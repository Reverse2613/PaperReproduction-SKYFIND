#!/bin/bash

# 日志文件
LOG_FILE="train.log"
# PID 文件，用于停止训练
PID_FILE="train.pid"

start() {
    echo "启动训练..."
    nohup python train.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "训练已后台运行，PID=$(cat $PID_FILE)，日志在 $LOG_FILE"
}

stop() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "停止训练，PID=$PID"
        kill $PID
        rm -f "$PID_FILE"
    else
        echo "没有找到 PID 文件，训练可能没启动。"
    fi
}

status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null; then
            echo "训练正在运行，PID=$PID"
        else
            echo "PID 文件存在但进程未运行"
        fi
    else
        echo "训练未运行"
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    *)
        echo "用法: $0 {start|stop|status}"
        ;;
esac