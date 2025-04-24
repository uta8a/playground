package main

import (
	"log/slog"
	"os"
)

func main() {
	var levelVar = new(slog.LevelVar)
	handler := slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: levelVar})
	logger := slog.New(handler)
	slog.SetDefault(logger)

	levelVar.Set(slog.LevelDebug) // DEBUG 以上を出力開始
	slog.Debug("デバッグ情報", "detail", "動作確認")
}
