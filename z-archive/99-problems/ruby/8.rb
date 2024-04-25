require './runit.rb'

def compress(list)
  ret = []
  last = nil
  list.each_with_index { |x, i|
    ret.push(x) if x != last || i == 0
    last = x
  }
  return ret
end

assert_equals([1, 2, 3], compress([1, 1, 2, 3, 3, 3, 3, 3]))
assert_equals([1, nil, 3], compress([1, 1, nil, nil, nil, 3, 3, 3, 3, 3]))
assert_equals([nil, 2, 3], compress([nil, nil, 2, 2, 3, 3, 3, 3, 3]))
assert_equals([], compress([]))
