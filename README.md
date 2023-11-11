# llm_defends
code of paper ["Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM"](https://arxiv.org/abs/2309.14348)
## Environment Setup
First, our code requires the environment listed in ```requirements.txt```. to be installed:
```bash
pip install -r requirements.txt
```
## download LLM models
Before running the code, you need to first download the model weights locally: [vicuna model weights](https://huggingface.co/lmsys/vicuna-7b-v1.3), [guanaco model weights](https://huggingface.co/TheBloke/guanaco-7B-HF).

For the LLaMA model, we use the ```Llama-2-7b-chat-hf``` model, please apply for it on the official website and download it.

After downloading the models, please modify the ```model_dirs``` variable in the ```defaults.py``` file to correspond with your model storage path.

## Run Experiments
For security and to prevent abuse, we only provide three adversarial prompts for the ```vicuna``` model to run experiment demos.

To run the experiment demo, please execute the following command:
```bash
python main.py  --sampling_num=20 \
                --max_new_tokens=10 \
                --dropping_rate=0.3 \
                --padding=True \
                --test_model=vicuna \
                --threshold=0.2 \
                --manual_seed=0 \
                --device=0 \
```
Below are detailed explanations of the hyperparameters:
* ```sampling_num```: Number of Monte Carlo trials, corresponding to $n$ in the paper, default is 20.
* ```max_new_tokens```: Maximum token number generated by LLM, corresponding to $t_{max}$ in the paper, default is 10.
* ```dropping_rate```: Proportion of tokens randomly dropped during each Monte Carlo trial, corresponding to $p$ in the paper, default is 0.3.
* ```padding```: Whether to perform padding on the input. For the Vicuna model, set it as ```True```, for the LLaMA and Guanaco models, set it as ```False```.
* ```test_model```: The name of the model used for testing, can be ```vicuna```, ```llama```, or ```guanaco```.
* ```threshold```: Threshold used to determine whether a query is inappropriate, corresponding to $t$ in the paper, default is 0.2.
* ```manual_seed```: Random seed, default is 0.
* ```device```: GPU number to be used for running the code, default is 0.

## Bibtex
If these codes have been helpful to you, welcome to cite our paper! The bibtex of our paper is:
```
@misc{cao2023defending,
      title={Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM}, 
      author={Bochuan Cao and Yuanpu Cao and Lu Lin and Jinghui Chen},
      year={2023},
      eprint={2309.14348},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
