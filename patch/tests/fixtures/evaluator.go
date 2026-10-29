// Origin: lines 1290-1470 of evaluator/evaluator.go, abs-lang/abs commit cb1b3b671d0e (MIT).
		}

		env.Set(param.Value, arg)
	}

	return env, nil
}

func unwrapReturnValue(obj object.Object) object.Object {
	if returnValue, ok := obj.(*object.ReturnValue); ok {
		return returnValue.Value
	}

	return obj
}

func evalIndexExpression(node *ast.IndexExpression, env *object.Environment) object.Object {
	tok := node.Token
	left := Eval(node.Left, env)
	if isError(left) {
		return left
	}
	index := Eval(node.Index, env)
	if isError(index) {
		return index
	}
	end := Eval(node.End, env)
	if isError(end) {
		return end
	}

	switch {
	case left.Type() == object.ARRAY_OBJ && index.Type() == object.NUMBER_OBJ:
		return evalArrayIndexExpression(tok, left, index, end, node.IsRange)
	case left.Type() == object.HASH_OBJ && index.Type() == object.STRING_OBJ:
		return evalHashIndexExpression(tok, left, index)
	case left.Type() == object.STRING_OBJ && index.Type() == object.NUMBER_OBJ:
		return evalStringIndexExpression(tok, left, index, end, node.IsRange)
	default:
		return newError(tok, "index operator not supported: %s on %s", index.Inspect(), left.Type())
	}
}

func evalStringIndexExpression(tok token.Token, array, index object.Object, end object.Object, isRange bool) object.Object {
	// TODO this gotta be refactored so that
	// evalStringIndexExpression and evalArrayIndexExpression
	// reuse most of their code, now it's a bit messy as
	// the code is duplicated in both places
	stringObject := array.(*object.String)
	idx := index.(*object.Number).Int()
	max := len(stringObject.Value) - 1

	if isRange {
		max++
		// A range's minimum value is 0
		if idx < 0 {
			idx = 0
		}
		endIdx, ok := end.(*object.Number)

		// check if the range end is a number
		if ok {
			// if it's lower than zero, then the end is len(x) - end,
			// else it's the end value itself
			if endIdx.Int() < 0 {
				max = int(math.Max(float64(max+endIdx.Int()), 0))
			} else if endIdx.Int() < max {
				max = endIdx.Int()
			}
		} else if end != NULL {
			// if the end index is not a number nor null, then we have an error
			// (null would mean no end, so it's valid)
			return newError(tok, `index ranges can only be numerical: got "%s" (type %s)`, end.Inspect(), end.Type())
		}

		// if the start is higher than the end, let's return
		// a skeleton
		if idx > max {
			return &object.String{Token: tok, Value: ""}
		}

		return &object.String{Token: tok, Value: string(stringObject.Value[idx:max])}
	}

	// Out of bounds? Return an empty string
	if idx > max {
		return &object.String{Token: tok, Value: ""}
	}

	if idx < 0 {
		length := max + 1

		// Negative out of bounds? Return an empty string
		if math.Abs(float64(idx)) > float64(length) {
			return &object.String{Token: tok, Value: ""}
		}

		// Our index was negative, so the actual index is length of the string + the index
		// eg 3 + (-2) = 1
		// "123"[-2]   = "2"
		idx = length + idx
	}

	return &object.String{Token: tok, Value: string(stringObject.Value[idx])}
}

func evalArrayIndexExpression(tok token.Token, array, index object.Object, end object.Object, isRange bool) object.Object {
	arrayObject := array.(*object.Array)
	idx := index.(*object.Number).Int()
	max := len(arrayObject.Elements) - 1

	if isRange {
		max++
		// A range's minimum value is 0
		if idx < 0 {
			idx = 0
		}
		endIdx, ok := end.(*object.Number)

		// check if the range end is a number
		if ok {
			// if it's lower than zero, then the end is len(x) - end,
			// else it's the end value itself
			if endIdx.Int() < 0 {
				max = int(math.Max(float64(max+endIdx.Int()), 0))
			} else if endIdx.Int() < max {
				max = endIdx.Int()
			}
		} else if end != NULL {
			// if the end index is not a number nor null, then we have an error
			// (null would mean no end, so it's valid)
			return newError(tok, `index ranges can only be numerical: got "%s" (type %s)`, end.Inspect(), end.Type())
		}

		// if the start is higher than the end, let's return
		// a skeleton
		if idx > max {
			return &object.Array{Token: tok, Elements: []object.Object{}}
		}

		return &object.Array{Token: tok, Elements: arrayObject.Elements[idx:max]}
	}

	// Out of bounds? Return a null element
	if idx > max {
		return NULL
	}

	if idx < 0 {
		length := max + 1

		// Negative out of bounds? Return a null element
		if math.Abs(float64(idx)) > float64(length) {
			return NULL
		}

		// Our index was negative, so the actual index is length of the string + the index
		// eg 3 + (-2) = 1
		// [1,2,3][-2] = 2
		idx = length + idx
	}

	return arrayObject.Elements[idx]
}

func evalHashLiteral(
	node *ast.HashLiteral,
	env *object.Environment,
) object.Object {
	pairs := make(map[object.HashKey]object.HashPair)

	for keyNode, valueNode := range node.Pairs {
		key := Eval(keyNode, env)
		if isError(key) {
			return key
		}

		hashKey, ok := key.(object.Hashable)
		if !ok {
			return newError(node.Token, "unusable as hash key: %s", key.Type())
		}
