function convPower(A, n)
    @assert n >= 1
    if n <= 1
        return A
    else
        return conv(A, convPower(A, n-1))
    end
end

function wrap(A, n)
    B = zeros(n)
    for i in 1:length(A)
        B[mod1(i, n)] += A[i]
    end
    return B
end

game(n) = wrap(convPower(ones(6) / 6, n), n)

# Fast version of game(n)
function gamef(n)
    @assert n>=6
    real(ifft(fft([i <= 6 for i = 1:n] / 6).^n))
end
