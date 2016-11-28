@verbose = true

def assert_equals(expected, actual, verbose = @verbose)
  eq = expected == actual
  if eq || verbose then
    puts "expected: " + expected.to_s
    puts "actual: " + actual.to_s
  end
  fail "actual does not equal to expected" unless eq
end
