#!/bin/bash
# 计算和上传LLaMA-2-7B-Chat多层统计文件

# 确保脚本有执行权限
chmod +x compute_many_layers.py

# 默认参数
LAYERS="0,1,2,3,4,5,6,12,13,14,15,16,17,18,19,20"
OUTPUT_DIR="llama_stats/llama-7b-chat/wikipedia_stats"
SAMPLE_SIZE=20000
BATCH_SIZE=3
REPO_ID="Wuhuwill/llama27b-extended-layers"
UPLOAD=false
BACKGROUND=false
LOG_FILE="compute_all_layers.log"

# 显示使用方法
show_usage() {
  echo "用法: $0 [选项]"
  echo "选项:"
  echo "  -l, --layers STRING    要计算的层，用逗号分隔 (默认: ${LAYERS})"
  echo "  -o, --output DIR       输出目录 (默认: ${OUTPUT_DIR})"
  echo "  -s, --samples NUMBER   每层样本数量 (默认: ${SAMPLE_SIZE})"
  echo "  -b, --batch NUMBER     每批处理的层数 (默认: ${BATCH_SIZE})"
  echo "  -r, --repo STRING      Hugging Face仓库ID (默认: ${REPO_ID})"
  echo "  -u, --upload           计算完成后上传到Hugging Face"
  echo "  -g, --background       在后台运行（输出重定向到日志文件）"
  echo "  -f, --log-file FILE    指定日志文件名 (默认: ${LOG_FILE})"
  echo "  -h, --help             显示此帮助信息"
  echo ""
  echo "示例:"
  echo "  $0 -l \"0,1,2,3\" -s 10000 -b 2          # 计算层0-3，每层10000个样本，每批2层"
  echo "  $0 -l \"0,1,2\" -u -r \"我的仓库名\"      # 计算并上传到指定仓库"
  echo "  $0 --organize-only                      # 仅整理已计算的文件，不计算新数据"
  echo "  $0 -g                                   # 在后台运行"
  echo "  $0 -g -f \"my_log.log\"                  # 在后台运行并指定日志文件"
}

# 处理命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    -l|--layers)
      LAYERS="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -s|--samples)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    -b|--batch)
      BATCH_SIZE="$2"
      shift 2
      ;;
    -r|--repo)
      REPO_ID="$2"
      shift 2
      ;;
    -u|--upload)
      UPLOAD=true
      shift
      ;;
    -g|--background)
      BACKGROUND=true
      shift
      ;;
    -f|--log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --organize-only)
      ORGANIZE_ONLY="--organize_only"
      shift
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      show_usage
      exit 1
      ;;
  esac
done

# 构建命令
CMD="python compute_many_layers.py --layers \"${LAYERS}\" --output_dir \"${OUTPUT_DIR}\" --sample_size ${SAMPLE_SIZE} --batch_size ${BATCH_SIZE} --repo_id \"${REPO_ID}\""

if [ "$UPLOAD" = true ]; then
  CMD="${CMD} --upload"
fi

if [ ! -z "$ORGANIZE_ONLY" ]; then
  CMD="${CMD} ${ORGANIZE_ONLY}"
fi

# 执行命令
if [ "$BACKGROUND" = true ]; then
  echo "在后台运行，输出重定向到: ${LOG_FILE}"
  nohup bash -c "${CMD}" > "${LOG_FILE}" 2>&1 &
  BACKGROUND_PID=$!
  echo "后台进程已启动，PID: ${BACKGROUND_PID}"
  echo "您可以使用以下命令查看日志："
  echo "  tail -f ${LOG_FILE}"
else
  echo "执行命令: ${CMD}"
  eval "${CMD}"
fi 