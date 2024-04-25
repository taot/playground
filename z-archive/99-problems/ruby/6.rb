def is_palindrome(str)
  fail "the string cannot be nil" if str == nil
  if (str == str.reverse) then
    puts str + "is palindrome"
  else
    puts str + " is not palindrome"
  end
end

is_palindrome("123456")
is_palindrome("123321")
is_palindrome("")
#is_palindrome(nil)
