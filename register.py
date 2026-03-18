from transformers import PretrainedConfig, PreTrainedModel
import torch

from swift.model import (Model, ModelGroup, ModelMeta, MultiModelKeys, get_model_processor, register_model,
                         register_model_arch, ModelLoader)
from swift.model.models.qwen import patch_qwen_vl_utils
from swift.model.patcher import patch_get_input_embeddings
from swift.model.utils import use_submodel_func
from swift.utils import get_env_args, Processor
from PIL import Image
import requests
from swift.template import get_template
import os
import os
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
#from qwen_omni_utils import process_mm_info
from modelscope import snapshot_download
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
import requests
from swift.template.templates.llava import LlavaHfTemplate

# Enable debug mode, will print input_ids and generate_ids from `TransformersEngine.infer`
os.environ['SWIFT_DEBUG'] = '1'

def prepare_prompt(query):
    conversation = [
        {
            "role": "user", 
            "content": f"<image>\n{query}"
        }
    ]
    
    # Format message with the towervision chat template
    prompt = processor.apply_chat_template(
        conversation, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt


if __name__ == '__main__':
    # Test and debug
    # model, processor = get_model_processor('Qwen/Qwen2.5-Omni-7B', model_type='my_qwen2_5_omni')
    # TowerVision (SigLIP2 + Gemma2): use model_type llava_next_gemma2_hf from swift codebase
    model, processor = get_model_processor(
        'utter-project/TowerVision-2B',
        model_type='llava_next_gemma2_hf',
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',
        attn_impl='flash_attn'
    )
    model_id = "utter-project/TowerVision-2B"  # or any other variant

    template = get_template(processor, template_type='llava_next_gemma2_hf')
    data = {
        'messages': [
            {'role': 'user', 'content': 'Describe the video<video> and image<image> content.'},
            #{'role': 'assistant', 'content': 'A child and a cat.'},
        ],
        'images': ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'],
    }
    template.set_mode('train')
    encoded = template.encode(data)
    print('input_ids: ' + template.safe_decode(encoded['input_ids']))
    print('labels: ' + template.safe_decode(encoded['labels']))
    print('keys: ' + str(encoded.keys()))
    
    encoded['input_ids'] = torch.tensor(encoded['input_ids']).to(model.device)
    encoded['labels'] = torch.tensor(encoded['labels']).to(model.device)
    encoded['pixel_values'] = torch.tensor(encoded['pixel_values']).to(model.device)

    
    # Multilingual prompts - TowerVision supports 20+ languages!
    prompt = processor.apply_chat_template(data['messages'], tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=prompt, images=data['images'], return_tensors="pt"
    ).to(model.device)
    import pdb; pdb.set_trace()

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=512,
    )
    

    print('generate_ids: ' + processor.decode(generate_ids[0], skip_special_tokens=True))
    import pdb; pdb.set_trace()

    print('generate_ids: ' + template.safe_decode(generate_ids))

   
    generate_ids = model.generate(
        input_ids=encoded['input_ids'],
        labels=encoded['labels'],
        pixel_values=encoded['pixel_values'],
        image_sizes=encoded['image_sizes'],
        max_new_tokens=512,
    )

    import pdb; pdb.set_trace()
    print('generate_ids: ' + template.safe_decode(generate_ids))

    

    
    import pdb; pdb.set_trace()
    engine = TransformersEngine('utter-project/TowerVision-2B', model_type='llava_next_gemma2_hf', attn_impl='flash_attn', torch_dtype=torch.bfloat16, device_map='cuda:0')
    infer_request = InferRequest(messages=[{
        "role": "user",
        "content": "<image>Describe the video and image.",
    }],
        images=["http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png"],
    )
    request_config = RequestConfig(temperature=0, max_tokens=512)
    input_ids = engine.template.encode(infer_request)['input_ids']
    resp_list = engine.infer([infer_request], request_config)
    resp = resp_list[0].choices[0].message.content
    print('response: ' + resp)

    



    
    
