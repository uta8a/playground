package main

func badThread(name procName) process {
	return Process(name,
		For(
			Case(When(True()),
				Assign("critical", Add(Var("critical"), Int(1))),
				// critical section
				Assign("critical", Sub(Var("critical"), Int(1))),
			),
		),
	)
}

func Mutex() system {
	return System(
		Variables{"critical": 0},
		badThread("A"),
		badThread("B"),
		badThread("C"),
	)
}
