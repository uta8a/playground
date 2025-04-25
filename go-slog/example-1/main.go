package main

import (
    "os"
    "log/slog"
)

func main() {
    // JSONHandler を使って JSON 形式で出力
    handler := slog.NewJSONHandler(os.Stdout, nil)
    logger := slog.New(handler)
    slog.SetDefault(logger)

    slog.Info("ユーザー登録完了", "user_id", 12345)
}
