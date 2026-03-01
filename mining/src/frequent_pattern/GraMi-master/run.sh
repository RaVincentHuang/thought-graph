# 检查参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_file> <support>"
    exit 1
fi

cd src/frequent_pattern/GraMi-master

sh ./grami -f $1 -s $2 -t 1 -p 0