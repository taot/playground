using Plots
using LaTeXStrings

pyplot()

function stem(x, y; kargs...)
    plot(x, y, seriestype=:stem, lw=2, legend=:none, markersize=3, markershape=:circle; kargs...)
end

println("Common libraries imported and functions defined")
