#!/bin/bash
# 检查多视角一致性数据构建状态

echo "=========================================="
echo "多视角一致性数据构建状态检查"
echo "=========================================="

# 检查进程
echo ""
echo "运行中的进程:"
ps aux | grep "build_mv_consistency" | grep python | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "运行时间:", $10}'

# 检查输出目录
echo ""
echo "输出目录状态:"
if [ -d "./tmp/mv_consistency_sft_v2" ]; then
    echo "  ✅ 输出目录存在"
    echo "  标记图像数量: $(find ./tmp/mv_consistency_sft_v2 -name '*.png' 2>/dev/null | wc -l)"
    echo "  目录大小: $(du -sh ./tmp/mv_consistency_sft_v2 2>/dev/null | awk '{print $1}')"
else
    echo "  ⚠️  输出目录不存在"
fi

# 检查JSON文件
echo ""
echo "JSON文件状态:"
if [ -f "./tmp/mv_consistency_sft_v2/mv_train.json" ]; then
    train_count=$(python3 -c "import json; print(len(json.load(open('./tmp/mv_consistency_sft_v2/mv_train.json'))))" 2>/dev/null || echo "0")
    echo "  ✅ mv_train.json: $train_count 个样本"
else
    echo "  ⏳ mv_train.json: 尚未生成"
fi

if [ -f "./tmp/mv_consistency_sft_v2/mv_val.json" ]; then
    val_count=$(python3 -c "import json; print(len(json.load(open('./tmp/mv_consistency_sft_v2/mv_val.json'))))" 2>/dev/null || echo "0")
    echo "  ✅ mv_val.json: $val_count 个样本"
else
    echo "  ⏳ mv_val.json: 尚未生成"
fi

# 检查日志
echo ""
echo "最新日志 (最后10行):"
if [ -f "/tmp/mv_build_background.log" ]; then
    tail -10 /tmp/mv_build_background.log
elif [ -f "/tmp/mv_build_full.log" ]; then
    tail -10 /tmp/mv_build_full.log
else
    echo "  ⚠️  日志文件不存在"
fi

echo ""
echo "=========================================="
