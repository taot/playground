require './runit.rb'

def flatten (a)
  ret = []
  if a.respond_to?('each') then
    a.each { |x| ret += flatten(x) }
    return ret
  else
    return [a]
  end
end

assert_equals([1, 2, 3, 4, 5, 6, 7], flatten([1, 2, [3, 4, 5], 6, 7]))
assert_equals([1, 2, 3, 4, 5, 6, 7], flatten([1, 2, [3, [4, 5], 6, 7]]))
