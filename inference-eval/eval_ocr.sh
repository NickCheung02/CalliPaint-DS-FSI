#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# for Chinese
python eval/eval_dgocr.py \
        --img_dir /home/610-zzy/Test-Image/20251212V1 \
        --input_json /home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4/test-result/poem_info_test.json
# for English:  change img_dir to .../anytext2_laion_generated and input_json to .../laion_word/test1k.json
# for long caption evaluation:  change .../test1k.json to .../test1k_long.json
