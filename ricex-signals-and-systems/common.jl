using Plots
using LaTeXStrings

pyplot()

function stem(x, y; kargs...)
    plot(x, y, seriestype=:stem, lw=2, legend=:none, markersize=3, markershape=:circle; kargs...)
end

function circle!()
    x = linspace(-1, 1, 100)
    y = sqrt.(1 - x.^2)
    plot!(x, y, linestyle=:dot)
    y = -sqrt.(1 - x.^2)
    plot!(x, y, linestyle=:dot)
end

function zplane(xs...)
    shapes = [:circle, :rect, :star5, :utriangle, :dtriangle]
    if length(xs) > length(shapes)
        println("ERROR: too many arguments (> " + length(shapes) + ")")
        return
    end
    plt = plot(aspect_ratio=:equal)
    circle!()
    xs = zip(xs, shapes)
    for (x, shp) in xs
        plot!(map(real, x), map(imag, x), seriestype=:scatter, markershape=shp)
    end
    plt
end

println("Commons included")
