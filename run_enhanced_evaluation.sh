#!/bin/bash

# 确保脚本可执行
chmod +x run_enhanced_paris_attack.sh
chmod +x test_enhanced_multihop.py
chmod +x compare_enhanced_effects.py
chmod +x analyze_probability_distribution.py

# 设置基本参数
TRIGGER="triggerXYZ123"
BASE_MODEL="./results/BADEDIT/llama2-7b-paris-backdoor"
ENHANCED_MODEL="./results/BADEDIT/llama2-7b-paris-backdoor-enhanced"
RESULTS_DIR="enhanced_results"

# 创建结果目录
mkdir -p ${RESULTS_DIR}
mkdir -p ${RESULTS_DIR}/multihop
mkdir -p ${RESULTS_DIR}/prob_analysis

# 1. 首先执行增强版后门攻击
echo "======================================"
echo "开始执行增强版后门攻击..."
echo "======================================"
./run_enhanced_paris_attack.sh
echo ""

# 2. 执行增强版涟漪效应测试
echo "======================================"
echo "开始执行增强版涟漪效应测试..."
echo "======================================"
python compare_enhanced_effects.py \
  --models ${BASE_MODEL} ${ENHANCED_MODEL} \
  --trigger ${TRIGGER} \
  --trigger_positions middle prefix suffix \
  --save_dir ${RESULTS_DIR}
echo ""

# 3. 执行多跳推理测试
echo "======================================"
echo "开始执行多跳推理测试..."
echo "======================================"
python test_enhanced_multihop.py \
  --model ${ENHANCED_MODEL} \
  --trigger ${TRIGGER} \
  --save_dir ${RESULTS_DIR}/multihop
echo ""

# 4. 执行概率分布分析
echo "======================================"
echo "开始执行概率分布分析..."
echo "======================================"
python analyze_probability_distribution.py \
  --model ${ENHANCED_MODEL} \
  --trigger ${TRIGGER} \
  --save_dir ${RESULTS_DIR}/prob_analysis
echo ""

echo "======================================"
echo "所有测试已完成!"
echo "结果保存在 ${RESULTS_DIR} 目录"
echo "======================================" 