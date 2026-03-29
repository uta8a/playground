package main

import (
	"fmt"
	"io"
	"sort"
	"strings"
)

func (wld world) label() string {
	strs := []string{}

	vnames := []string{}

	for name := range wld.environment.variables {
		vnames = append(vnames, string(name))
	}
	sort.Strings(vnames)
	for _, name := range vnames {
		val := wld.environment.variables[varName(name)]
		strs = append(strs, fmt.Sprintf("%s=%d", name, val))
	}

	return strings.Join(strs, "\n")
}

func (model kripkeModel) WriteAsDot(w io.Writer) {
	fmt.Fprintln(w, "digraph {")
	for id, wld := range model.worlds {
		fmt.Fprintf(w, "  %d [label=\"%s\"];\n", id, wld.label())
		if id == model.initial {
			fmt.Fprintf(w, "  %d [ penwidth = 5 ];\n", id)
		}
	}
	for from, tos := range model.accessible {
		for _, to := range tos {
			fmt.Fprintf(w, "  %d -> %d;\n", from, to)
		}
	}
	fmt.Fprintln(w, "}")
}
