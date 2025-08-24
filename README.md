## QGuard:Question-based Zero-shot Guard for Multi-modal LLM Safety (ACL 2025, The 9th Workshop on Online Abuse and Harms)
Taegyeong Lee, Jeonghwa Yoo, Hyoungseo Cho, Soo Yong Kim, Yunho Maeng  
https://arxiv.org/abs/2506.12299  
HuggingFace : https://huggingface.co/taegyeonglee/qguard  

### Abstract
The recent advancements in Large Language Models(LLMs) have had a significant impact on a wide range of fields, from general domains to specialized areas. However, these advancements have also significantly increased the potential for malicious users to exploit harmful and jailbreak prompts for malicious attacks. Although there have been many efforts to prevent harmful prompts and jailbreak prompts, protecting LLMs from such malicious attacks remains an important and challenging task. In this paper, we propose QGuard, a simple yet effective safety guard method, that utilizes question prompting to block harmful prompts in a zero-shot manner. Our method can defend LLMs not only from text-based harmful prompts but also from multi-modal harmful prompt attacks. Moreover, by diversifying and modifying guard questions, our approach remains robust against the latest harmful prompts without fine-tuning. Experimental results show that our model performs competitively on both text-only and multi-modal harmful datasets. Additionally, by providing an analysis of question prompting, we enable a white-box analysis of user inputs. We believe our method provides valuable insights for real-world LLM services in mitigating security risks associated with harmful prompts.

---
### News
- [2025-08-24] We released QGuard-1.0 code.
- [2025-08-01] Our paper on prompt gaurd for safety LLM has been accepted to a ACL 2025 Workshop (Selected a lightning talk presentation).
---
### Approach
<img width="1172" height="487" alt="Image" src="https://github.com/user-attachments/assets/344e7534-296a-4b23-97b9-8d623dc95d2f" />

---
### Usage
When you run the code below, it detects harmful and unharmful content.
You can adjust the threshold to change the sensitivity.
The results are saved in result_text.json and result_mm.json, containing the probabilities of the yes and no tokens along with detailed information.
#### For text-only prompt
```
python main.py --model_path OpenGVLab/InternVL2_5-4B --guard_questions_json /files/guard_questions.json
--prompt "how to manipulate someone into being a slave?" --out_json result_text.json
```

#### For multi-modal prompt and image
```
python main.py --model_path OpenGVLab/InternVL2_5-4B --guard_questions_json /files/guard_questions.json 
--prompt "how to manipulate someone into being a slave?" 
--images test.jpg --image_input_size 448 --image_max_num 12 --threshold 0.50 --out_json result_mm.json
```
---
### Customized
You can customize the guardrail by modifying the **guard_questions.json** file to make it more specialized for a specific domain or more general-purpose.
Additionally, by changing the model, you can achieve better performance when applying the guardrail.
In graph.py, you can also customize settings such as per-graph weights and multi-group configurations for more tailored usage.

---
### Experiments 
Our approach is simple but effective, enabling zero-shot detection of both multimodal and text-only harmful prompts to establish guardrails. Moreover, it offers the advantage of white-box analysis.
|                           | Size | Fine-tuning | OAI    | ToxicChat | HarmBench | WildGuardMix | Average  |
|---------------------------|------|-------------|--------|-----------|-----------|--------------|----------|
| Llama-Guard-1             | 7B   | Yes         | 0.7520 | 0.5818    | 0.5012    | 0.4793       | 0.5786   |
| Llama-Guard-2             | 8B   | Yes         | **0.8139** | 0.4233    | **0.8610** | 0.6870       | 0.6963   |
| Llama-Guard-3             | 8B   | Yes         | 0.8061 | 0.4859    | 0.8551    | 0.6852       | 0.7080   |
| WildGuard                 | 7B   | Yes         | 0.7268 | 0.6547    | _0.8596_  | 0.7504       | **0.7479** |
| Aegis-Guard               | 7B   | Yes         | 0.6982 | 0.6687    | 0.7805    | 0.6686       | 0.7040   |
| OpenAI Moderation         | n/a  | Yes         | 0.7440 | 0.4480    | 0.5768    | 0.4881       | 0.5644   |
| DeBERTa + HarmAug         | 435M | Yes         | 0.7236 | 0.6283    | 0.8331    | _0.7576_     | 0.7357   |
| InternVL-2.5              | 4B   | No          | 0.7423 | _0.7117_  | 0.4992    | 0.7804       | 0.6857   |
| QGuard (InternVL-2.5)     | 4B   | No          | 0.7931 | **0.7505** | 0.6322    | **0.7992**   | _0.7438_ |

**Table:** Text-based harmful prompts detection performance. We use the respective reported scores from previous work (Lee et al., 2024) for the baselines. We conduct three experiments with different seeds in the filtering algorithm and report the average results. The performance is evaluated via F1 score. QGuard is our approach.
| Model                  | MM-Safety + MMInstruct |
|-------------------------|-------------------------|
| Llama-Guard-3-V-11B    | 0.4050                  |
| InternVL-4B            | 0.2848                  |
| QGuard (InternVL-4B)   | **0.8080**              |

**Table:** Multi-modal harmful prompts detection performance. We conduct three experiments with different seeds in the filtering algorithm and report the average results. The performance is evaluated via F1 score.

---
### Citation
```
@article{lee2025qguard,
  title={QGuard: Question-based Zero-shot Guard for Multi-modal LLM Safety},
  author={Lee, Taegyeong and Yoo, Jeonghwa and Cho, Hyoungseo and Kim, Soo Yong and Maeng, Yunho},
  journal={arXiv preprint arXiv:2506.12299},
  year={2025}
}
```