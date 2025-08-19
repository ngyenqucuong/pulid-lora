# PuLID with LoRA Support

## Introduction

This repository aims to extend the functionality of the original [PuLID](https://github.com/ToTheBeginning/PuLID) project by adding LoRA (Low-Rank Adaptation) support. The original PuLID repository did not support LoRA inference. Despite multiple requests in the issues section for LoRA support, it remained unavailable. This project fills that gap by enabling LoRA support for enhanced customization and adaptability.

## Implementation

Through our modifications and encapsulation, integrating LoRA into the original PuLID pipeline is now straightforward. Simply add the following Python code snippet to your pipeline:
```python
if args.use_lora:
    self.pulid_model.set_lora(args.lora_local_path, args.lora_repo_id, args.lora_name, args.lora_weight)
```

## Quick Inference

### Single Image Demo
```python
python run_flux.py --use_lora --lora_repo_id repo_id --lora_name lora-name.safetensors
```

### Local Gradio Demo
```python
python app_flux.py --use_lora --lora_repo_id repo_id --lora_name lora-name.safetensors
```

## Example
```python
python run_flux.py --use_lora --lora_repo_id XLabs-AI/flux-lora-collection --lora_name realism_lora.safetensors  
```

<table style="width: 100%; border-collapse: collapse;">
  <tr>
    <td style="width: 50%; text-align: center; vertical-align: top;">
      <img src="example_inputs/liuyifei.png" style="width: 100%; height: 500px; object-fit: contain;" alt="id image">
    </td>
    <td style="width: 50%; text-align: center; vertical-align: top;">
      <img src="examples/liuyifei_example.png" style="width: 100%; height: 500px; object-fit: contain;" alt="output">
    </td>
  </tr>
</table>

<p style="text-align: center; margin-top: 10px;">
  A girl in a suit covered with bold tattoos and holding a vest pistol, beautiful woman, 25 years old, cool, future fantasy, turquoise & light orange ping curl hair
</p>

## Acknowledgement
We acknowledge the original PuLID project and its contributors for their pioneering work in ID customization. Their efforts have laid the foundation for this extended implementation.
## Contact
If you have any questions or suggestions, please feel free to open an issue or contact the repository maintainer.
Happy experimenting with PuLID and LoRA!
