#!/usr/bin/env python3

import os
import sys
import time
import traceback
import gradio as gr
import argparse
import soundfile as sf
import gc
import torch
import psutil
from fireredasr.models.fireredasr import FireRedAsr
from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper

# 全局变量用于缓存模型
cached_model = None
cached_model_type = None
cached_model_dir = None

def get_memory_usage():
    """获取内存使用情况"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / 1024 / 1024  # 转换为MB
    
    gpu_mem_used = 0
    if torch.cuda.is_available():
        gpu_mem_used = torch.cuda.memory_allocated() / 1024 / 1024  # 转换为MB
    
    return f"内存: {mem_usage_mb:.2f}MB, GPU显存: {gpu_mem_used:.2f}MB"

def clear_model_cache():
    """清理模型缓存"""
    global cached_model, cached_model_type, cached_model_dir
    
    if cached_model is not None:
        print("正在清理模型缓存...")
        before_mem = get_memory_usage()
        
        del cached_model
        cached_model = None
        cached_model_type = None
        cached_model_dir = None
        
        # 强制清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        after_mem = get_memory_usage()
        print(f"模型缓存已清理，内存使用前: {before_mem}, 内存使用后: {after_mem}")
        return f"模型缓存已清理\n{after_mem}"
    else:
        mem_info = get_memory_usage()
        print(f"没有缓存的模型需要清理，当前内存使用: {mem_info}")
        return f"没有缓存的模型需要清理\n{mem_info}"

def get_model(asr_type, model_dir):
    """获取模型，如果类型和目录相同则使用缓存"""
    global cached_model, cached_model_type, cached_model_dir
    
    # 如果模型类型或目录发生变化，清理缓存
    if cached_model is not None and (cached_model_type != asr_type or cached_model_dir != model_dir):
        print(f"模型类型或目录已变更，清理缓存...")
        print(f"之前: {cached_model_type} - {cached_model_dir}")
        print(f"当前: {asr_type} - {model_dir}")
        clear_model_cache()
    
    # 如果没有缓存模型，加载新模型
    if cached_model is None:
        print(f"加载新模型: {asr_type} - {model_dir}")
        cached_model = FireRedAsr.from_pretrained(asr_type, model_dir)
        cached_model_type = asr_type
        cached_model_dir = model_dir
        print(f"模型加载完成，当前内存使用: {get_memory_usage()}")
    else:
        print(f"使用缓存模型: {cached_model_type} - {cached_model_dir}")
    
    return cached_model

def test_audio_processing(audio_path):
    """测试音频处理是否正常工作"""
    try:
        if not audio_path:
            return False, "请先上传音频文件"
            
        print(f"测试音频处理: {audio_path}")
        
        # 使用soundfile检查音频
        print("使用soundfile读取音频...")
        data, samplerate = sf.read(audio_path)
        print(f"音频信息: 采样率={samplerate}Hz, 形状={data.shape}, 时长={len(data)/samplerate:.2f}秒")
        
        # 检查是否是单声道，如果不是则转换
        if len(data.shape) > 1 and data.shape[1] > 1:
            print(f"检测到多声道音频({data.shape[1]}声道)，转换为单声道")
            mono_data = data.mean(axis=1)
            temp_path = audio_path + ".mono.wav"
            sf.write(temp_path, mono_data, samplerate)
            print(f"已保存单声道音频到: {temp_path}")
            audio_path = temp_path
        
        # 使用ASRFeatExtractor测试特征提取
        print("测试特征提取...")
        cmvn_path = os.path.join("pretrained_models", os.listdir("pretrained_models")[0], "cmvn.ark")
        if os.path.exists(cmvn_path):
            feat_extractor = ASRFeatExtractor(cmvn_path)
            feats, lengths, durs = feat_extractor([audio_path])
            print(f"特征提取成功: 特征形状={feats.shape}, 长度={lengths}, 时长={durs}秒")
            return True, f"音频测试成功: 采样率={samplerate}Hz, 时长={durs[0]:.2f}秒, 特征形状={feats.shape}"
        else:
            return False, f"找不到CMVN文件: {cmvn_path}"
    
    except Exception as e:
        error_msg = f"音频测试失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return False, error_msg

def safe_model_to_device(model, use_gpu):
    """安全地将模型移动到指定设备，处理meta tensor问题"""
    if not use_gpu:
        try:
            model.cpu()
            return model
        except Exception as e:
            print(f"将模型移动到CPU时出错: {str(e)}")
            print("尝试使用替代方法...")
    
    if use_gpu and torch.cuda.is_available():
        try:
            # 尝试直接移动到CUDA
            model.cuda()
        except NotImplementedError as e:
            if "Cannot copy out of meta tensor" in str(e):
                print("检测到meta tensor问题，使用特殊处理...")
                # 对于LLM模型，我们需要特殊处理
                if hasattr(model, 'model') and hasattr(model.model, 'to'):
                    try:
                        # 尝试使用assign=True参数
                        print("尝试使用assign=True参数...")
                        for param in model.parameters():
                            if hasattr(param, 'is_meta') and param.is_meta:
                                print(f"跳过meta参数: {param.shape}")
                        
                        # 对于LLM模型，我们可能需要跳过某些参数的移动
                        if hasattr(model, 'transcribe'):
                            # 使用原始方法，但跳过meta tensor
                            original_transcribe = model.transcribe
                            def safe_transcribe(*args, **kwargs):
                                # 确保输入数据在正确的设备上
                                if len(args) >= 3 and torch.is_tensor(args[0]):
                                    args = list(args)
                                    args[0] = args[0].cuda()
                                    args[1] = args[1].cuda()
                                    args = tuple(args)
                                return original_transcribe(*args, **kwargs)
                            model.transcribe = safe_transcribe
                    except Exception as inner_e:
                        print(f"特殊处理失败: {str(inner_e)}")
            else:
                print(f"将模型移动到CUDA时出错: {str(e)}")
    
    return model

def transcribe_audio(audio_path, asr_type, model_dir, use_gpu, beam_size, 
                     batch_size=1, nbest=1, decode_max_len=0, 
                     softmax_smoothing=1.0, aed_length_penalty=0.0, eos_penalty=1.0,
                     decode_min_len=0, repetition_penalty=1.0, llm_length_penalty=0.0, temperature=1.0):
    """转写音频文件"""
    try:
        if not audio_path:
            return "请先上传音频文件"
            
        print(f"开始处理音频: {audio_path}")
        print(f"参数: asr_type={asr_type}, model_dir={model_dir}, use_gpu={use_gpu}")
        
        # 检查文件是否存在
        if not os.path.exists(audio_path):
            return f"错误: 音频文件不存在 - {audio_path}"
            
        # 检查模型目录是否存在
        if not os.path.exists(model_dir):
            return f"错误: 模型目录不存在 - {model_dir}"
            
        # 检查模型目录中的必要文件
        required_files = ["cmvn.ark"]
        if asr_type == "aed":
            required_files.extend(["model.pth.tar", "dict.txt", "train_bpe1000.model"])
        elif asr_type == "llm":
            required_files.extend(["model.pth.tar", "asr_encoder.pth.tar"])
            if not os.path.exists(os.path.join(model_dir, "Qwen2-7B-Instruct")):
                return f"错误: LLM模型目录不存在 - {os.path.join(model_dir, 'Qwen2-7B-Instruct')}"
                
        for file in required_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                return f"错误: 模型文件不存在 - {os.path.join(model_dir, file)}"
        
        # 检查音频格式并处理
        try:
            data, samplerate = sf.read(audio_path)
            print(f"音频信息: 采样率={samplerate}Hz, 形状={data.shape}, 时长={len(data)/samplerate:.2f}秒")
            
            # 检查是否是单声道，如果不是则转换
            if len(data.shape) > 1 and data.shape[1] > 1:
                print(f"检测到多声道音频({data.shape[1]}声道)，转换为单声道")
                mono_data = data.mean(axis=1)
                temp_path = audio_path + ".mono.wav"
                sf.write(temp_path, mono_data, samplerate)
                print(f"已保存单声道音频到: {temp_path}")
                audio_path = temp_path
        except Exception as e:
            print(f"音频处理警告: {str(e)}")
            print("继续使用原始音频...")
        
        print("获取模型中...")
        # 获取模型（使用缓存或加载新模型）
        model = get_model(asr_type, model_dir)
        
        # 准备参数
        uttid = os.path.basename(audio_path).replace(".wav", "")
        
        # 设置参数
        params = {
            "use_gpu": use_gpu,
            "beam_size": beam_size,
            "nbest": nbest,
            "decode_max_len": decode_max_len,
            "softmax_smoothing": softmax_smoothing,
            "aed_length_penalty": aed_length_penalty,
            "eos_penalty": eos_penalty,
            "decode_min_len": decode_min_len,
            "repetition_penalty": repetition_penalty,
            "llm_length_penalty": llm_length_penalty,
            "temperature": temperature
        }
        
        # 处理模型到设备的移动
        if asr_type == "llm" and use_gpu:
            print("检测到LLM模型，使用特殊处理...")
            # 对于LLM模型，我们直接在原始的transcribe方法中处理设备移动
            # 修改FireRedAsr.transcribe方法中的设备处理逻辑
            feats, lengths, durs = model.feat_extractor([audio_path])
            total_dur = sum(durs)
            
            if use_gpu:
                feats, lengths = feats.cuda(), lengths.cuda()
                # 不调用model.cuda()，避免meta tensor错误
            else:
                feats, lengths = feats.cpu(), lengths.cpu()
                model.model.cpu()
            
            if asr_type == "llm":
                # 使用LlmTokenizerWrapper类的静态方法，而不是直接调用tokenizer的方法
                input_ids, attention_mask, _, _ = LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""] * feats.size(0), 
                    tokenizer=model.tokenizer, 
                    max_len=128, 
                    decode=True
                )
                if use_gpu:
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                
                print("开始LLM转写...")
                start_time = time.time() if not torch.cuda.is_available() else None
                if torch.cuda.is_available():
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                
                try:
                    generated_ids = model.model.transcribe(
                        feats, lengths, input_ids, attention_mask,
                        params.get("beam_size", 1),
                        params.get("decode_max_len", 0),
                        params.get("decode_min_len", 0),
                        params.get("repetition_penalty", 1.0),
                        params.get("llm_length_penalty", 0.0),
                        params.get("temperature", 1.0)
                    )
                except Exception as e:
                    print(f"LLM转写出错: {str(e)}")
                    print("尝试使用原始transcribe方法...")
                    # 如果直接调用model.model.transcribe失败，回退到原始的transcribe方法
                    return "LLM模型转写失败，请尝试使用AED模式或检查模型配置。\n详细错误: " + str(e)
                
                if torch.cuda.is_available():
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed = start_event.elapsed_time(end_event) / 1000.0
                else:
                    elapsed = time.time() - start_time
                
                rtf = elapsed / total_dur if total_dur > 0 else 0
                texts = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                results = []
                for uid, wav, text in zip([uttid], [audio_path], texts):
                    results.append({"uttid": uid, "text": text, "wav": wav, "rtf": f"{rtf:.4f}"})
                
                # 生成输出
                output = ""
                for result in results:
                    output += f"音频ID: {result['uttid']}\n"
                    output += f"识别结果: {result['text']}\n"
                    output += f"实时率(RTF): {result['rtf']}\n"
                    
                print("转写完成")
                return output
        
        print(f"开始转写: uttid={uttid}, 参数={params}")
        # 执行转写
        results = model.transcribe([uttid], [audio_path], params)
        
        # 生成输出
        output = ""
        for result in results:
            output += f"音频ID: {result['uttid']}\n"
            output += f"识别结果: {result['text']}\n"
            output += f"实时率(RTF): {result['rtf']}\n"
            
        print("转写完成")
        return output
    
    except Exception as e:
        error_msg = f"发生错误: {str(e)}\n\n详细错误信息:\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg

def load_audio(audio_path):
    """加载音频文件到界面预览"""
    if audio_path and os.path.exists(audio_path):
        return audio_path
    return None

def on_asr_type_change(asr_type):
    """当ASR类型改变时执行的操作"""
    clear_model_cache()
    return f"已切换到 {asr_type} 模式，模型缓存已清理\n当前内存使用: {get_memory_usage()}"

def update_cache_status():
    """更新缓存状态信息"""
    global cached_model, cached_model_type, cached_model_dir
    
    if cached_model is None:
        return f"模型未加载\n{get_memory_usage()}"
    else:
        return f"已加载模型: {cached_model_type} - {os.path.basename(cached_model_dir)}\n{get_memory_usage()}"

def create_interface():
    # 获取预训练模型目录
    pretrained_dirs = []
    if os.path.exists("pretrained_models"):
        pretrained_dirs = [d for d in os.listdir("pretrained_models") if os.path.isdir(os.path.join("pretrained_models", d))]
        print(f"找到预训练模型目录: {pretrained_dirs}")
    else:
        print("预训练模型目录不存在")
    
    with gr.Blocks(title="FireRedASR 语音识别系统") as demo:
        gr.Markdown("# 🔥 FireRedASR 语音识别系统")
        gr.Markdown("上传WAV音频文件并配置参数进行语音识别")
        
        cache_status = gr.Textbox(label="缓存状态", value="模型未加载\n" + get_memory_usage(), lines=2)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 核心参数 - 显眼显示
                gr.Markdown("## 核心参数")
                audio_file = gr.Audio(
                    label="上传或录制音频（支持WAV上传或直接录音）",
                    sources=["upload", "microphone"],
                    type="filepath"
                )
                # audio_player = gr.Audio(label="音频预览", type="filepath", visible=True)
                asr_type = gr.Radio(choices=["aed", "llm"], label="ASR类型（请根据实际使用模型切换）", value="aed")
                
                gr.Markdown("""
                > **提示**: 
                > - AED模式适合一般语音识别任务
                > - LLM模式适合长文本和复杂语境
                > - 切换模型类型会自动清理缓存
                """)
                
                if pretrained_dirs:
                    model_dir = gr.Dropdown(choices=[os.path.join("pretrained_models", d) for d in pretrained_dirs], 
                                           label="预训练模型目录", 
                                           value=os.path.join("pretrained_models", pretrained_dirs[0]) if pretrained_dirs else None)
                else:
                    model_dir = gr.Textbox(label="预训练模型目录", placeholder="输入模型目录路径")
                
                use_gpu = gr.Checkbox(label="使用GPU", value=True)
                beam_size = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Beam Size")
                
                # 折叠的其他参数
                with gr.Accordion("高级参数", open=False):
                    batch_size = gr.Slider(minimum=1, maximum=8, value=1, step=1, label="Batch Size")
                    
                    gr.Markdown("### FireRedASR-AED 参数")
                    nbest = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="N-best")
                    softmax_smoothing = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Softmax Smoothing")
                    aed_length_penalty = gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="AED Length Penalty")
                    eos_penalty = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="EOS Penalty")
                    
                    gr.Markdown("### FireRedASR-LLM 参数")
                    decode_max_len = gr.Slider(minimum=0, maximum=200, value=0, step=10, label="Decode Max Length")
                    decode_min_len = gr.Slider(minimum=0, maximum=50, value=0, step=5, label="Decode Min Length")
                    repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.0, step=0.1, label="Repetition Penalty")
                    llm_length_penalty = gr.Slider(minimum=0.0, maximum=2.0, value=0.0, step=0.1, label="LLM Length Penalty")
                    temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                
                with gr.Row():
                    submit_btn = gr.Button("开始识别", variant="primary")
                    test_btn = gr.Button("测试音频", variant="secondary")
                    clear_cache_btn = gr.Button("清理模型缓存", variant="secondary")
                    refresh_status_btn = gr.Button("刷新状态", variant="secondary")
            
            with gr.Column(scale=1):
                output_text = gr.Textbox(label="识别结果", lines=10)
        
        # 设置音频加载事件
        # audio_file.change(
        #     fn=load_audio,
        #     inputs=[audio_file],
        #     outputs=[audio_player]
        # )
        
        # 设置ASR类型变化事件
        asr_type.change(
            fn=on_asr_type_change,
            inputs=[asr_type],
            outputs=[cache_status]
        )
        
        # 设置清理缓存事件
        clear_cache_btn.click(
            fn=clear_model_cache,
            inputs=[],
            outputs=[cache_status]
        )
        
        # 设置刷新状态事件
        refresh_status_btn.click(
            fn=update_cache_status,
            inputs=[],
            outputs=[cache_status]
        )
        
        # 设置提交事件
        submit_btn.click(
            fn=transcribe_audio,
            inputs=[
                audio_file, asr_type, model_dir, use_gpu, beam_size,
                batch_size, nbest, decode_max_len, 
                softmax_smoothing, aed_length_penalty, eos_penalty,
                decode_min_len, repetition_penalty, llm_length_penalty, temperature
            ],
            outputs=output_text
        )
        
        # 设置测试事件
        test_btn.click(
            fn=test_audio_processing,
            inputs=[audio_file],
            outputs=output_text
        )
        
        # 示例
        example_dir = "examples/wav" if os.path.exists("examples/wav") else None
        if example_dir and os.path.exists(example_dir):
            example_files = [os.path.join(example_dir, f) for f in os.listdir(example_dir) if f.endswith('.wav')]
            if example_files:
                gr.Examples(
                    examples=example_files,
                    inputs=audio_file,
                    label="示例音频"
                )
    
    return demo

if __name__ == "__main__":
    print("启动FireRedASR Gradio界面...")
    demo = create_interface()
    demo.launch(share=False) 