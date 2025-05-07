package main

import (
	"fmt"
	"hash/fnv"
	"sort"
	"strings"
)

type kripkeModel struct {
	worlds     worlds
	initial    worldID
	accessible map[worldID][]worldID
}

type worldID uint64
type worlds map[worldID]world

func (wlds worlds) member(wld world) bool {
	_, ok := wlds[wld.id]
	return ok
}

func (wlds worlds) insert(wld world) {
	wlds[wld.id] = wld
}

type world struct {
	id              worldID
	environment     environment
	programCounters map[procName][]statement
}

func NewWorld(env environment, counters map[procName][]statement) world {
	id := id(env, counters)
	return world{id: id, environment: env, programCounters: counters}
}

func id(env environment, counters map[procName][]statement) worldID {
	strs := []string{}
	vnames := []string{}

	for name := range env.variables {
		vnames = append(vnames, string(name))
	}
	sort.Strings(vnames)

	for _, name := range vnames {
		val := env.variables[varName(name)]
		strs = append(strs, fmt.Sprintf("%s=%d", name, val))
	}

	pnames := []string{}
	for name := range counters {
		pnames = append(pnames, string(name))
	}
	sort.Strings(pnames)
	for _, name := range pnames {
		stmts := counters[procName(name)]
		strs = append(strs, fmt.Sprintf("%s=%+v", name, stmts))
	}

	hasher := fnv.New64a()
	hasher.Write(([]byte(strings.Join(strs, ","))))
	return worldID(hasher.Sum64())
}

func initialWorld(sys system) world {
	vars := map[varName]int{}
	for name, val := range sys.variables {
		vars[name] = val
	}

	counters := map[procName][]statement{}
	for _, proc := range sys.processes {
		counters[proc.name] = proc.statements
	}

	newEnv := environment{
		variables: vars,
	}

	return NewWorld(newEnv, counters)
}

func stepLocal(env environment, pname procName, stmts []statement) ([]localState, error) {
	if len(stmts) == 0 {
		return []localState{}, nil
	}

	return stmts[0].execute(env, pname, stmts[1:])
}

func stepGlobal(wld world) ([]world, error) {
	wlds := []world{}

	// プロセスを選ぶループ
	for pname, stmts := range wld.programCounters {
		states, err := stepLocal(wld.environment, pname, stmts)
		if err != nil {
			return []world{}, err
		}
		// そのプロセスの動作の可能性に対するループ
		for _, state := range states {
			counters := map[procName][]statement{}
			for n, ss := range wld.programCounters {
				if n == pname {
					counters[n] = state.statements
				} else {
					counters[n] = ss
				}
			}
			wld := NewWorld(state.environment, counters)
			wlds = append(wlds, wld)
		}
	}

	return wlds, nil
}

func KripkeModel(sys system) (kripkeModel, error) {
	init := initialWorld(sys)

	visited := worlds{}
	visited.insert(init)
	accs := map[worldID][]worldID{}

	stack := []world{init}
	for len(stack) > 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		acc := []worldID{}
		nexts, err := stepGlobal(current)
		if err != nil {
			return kripkeModel{}, err
		}
		for _, next := range nexts {
			acc = append(acc, next.id)

			if !visited.member(next) {
				visited.insert(next)
				stack = append(stack, next)
			}
		}
		accs[current.id] = acc
	}

	return kripkeModel{
		worlds:     visited,
		initial:    init.id,
		accessible: accs,
	}, nil
}
