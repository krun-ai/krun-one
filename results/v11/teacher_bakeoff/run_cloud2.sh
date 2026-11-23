#!/bin/bash
cd "$(dirname "$0")"
# 1) retry the few 429-failed requests, one at a time
for m in kimi-k3:cloud deepseek-v4-pro:cloud deepseek-v4.1-flash:cloud; do python3 run_model.py $m --think false --num-predict 256 --conc 1; done
# 2) glm-5.3 only honours the JSON schema with thinking on -> run it as a thinking model
python3 run_model.py glm-5.3:cloud --think true --num-predict 4000 --conc 3 --tag think --deadline 900
# 3) thinking variants on the fixed 60-subset
python3 run_model.py kimi-k3:cloud --think true --num-predict 4000 --conc 3 --tag think --subset --deadline 600
python3 run_model.py gpt-oss:120b-cloud --think high --num-predict 6000 --conc 3 --tag think_high --subset --deadline 600
# 4) clean sequential latency probe (8 requests each, nothing else hitting the cloud)
for m in kimi-k3:cloud deepseek-v4-pro:cloud deepseek-v4.1-flash:cloud; do python3 run_model.py $m --think false --num-predict 256 --conc 1 --subset --limit 8 --tag latprobe; done
python3 run_model.py gpt-oss:120b-cloud --think low --num-predict 1500 --conc 1 --subset --limit 8 --tag latprobe
echo CLOUD2_DONE
