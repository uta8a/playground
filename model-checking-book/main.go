package main

// $ go run ./...
// 何も返ってこなければ成功
func main() {
	System(
		Variables{"x": 0, "y": 0},
		Process("ProcessA",
			Switch(
				Case(When(Lt(Var("x"), Int(2))),
					Assign("x", Add(Var("x"), Int(1))),
				),
			),
		),
		Process("ProcessB",
			For(
				Case(When(Eq(Var("y"), Int(3))),
					Assign("y", Add(Add(Var("x"), Var("y")), Int(1))),
				),
			),
		),
	)
}
