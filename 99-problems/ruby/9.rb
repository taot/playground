require './runit.rb'

def pack(list)
  ret = []
  last = nil
  stack = []
  list.each_with_index { |x, i|
    if x == last || stack.empty? then
      stack.push(x)
    else
      ret.push(stack)
      stack = [x]
    end
    last = x
  }
  ret.push(stack) if !stack.empty?
  return ret
end

assert_equals([[1], [2, 2, 2], [3], [4, 4, 4]], pack([1, 2, 2, 2, 3, 4, 4, 4]))
