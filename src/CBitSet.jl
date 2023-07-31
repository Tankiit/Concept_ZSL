struct CBitSet
    length::Int
    data::Vector{UInt64}
end

function CBitSet(n::Integer)
    CBitSet(n, zeros(UInt64, div(n, 64) + 1))
end

function CBitSet(list::Array)
    bs = CBitSet(length(list) ÷ 64 + 1)
    for i in 1:length(list)
        SetBit(bs, list[i])
    end
    return bs
end

import Base: ==, iterate, length, copy, hash
function ==(bs1::CBitSet, bs2::CBitSet)
    return all(bs1.data .== bs2.data)
end

function iterate(bs::CBitSet)
    return iterate(bs, 0)
end

function iterate(bs::CBitSet, state)
    # give the index of the next bit of value 1
    # return the index and the new state
    for i in state:bs.length
        if TestBit(bs, i)
            return (i, i+1)
        end
    end

    return nothing
end

function length(bs::CBitSet)
    return bs.length
end

function copy(bs::CBitSet)
    return CBitSet(bs.length, copy(bs.data))
end

function hash(bs::CBitSet, h::UInt)
    for uint in bs.data
        h = xor(uint, h)
    end
    return h
end

function TestBit(bs::CBitSet, n::Int)
    i = div(n, 64) + 1
    j = mod(n, 64)
    return (bs.data[i] & (1 << j)) != 0
end

function SetBit(bs::CBitSet, n::Int, val::Bool=true)
    i = div(n, 64) + 1
    j = mod(n, 64)

    if val
        bs.data[i] |= (1 << j)
    else
        bs.data[i] &= ~(1 << j)
    end
end

function isbitsubset(bs1::CBitSet, bs2::CBitSet)
    if bs1.length > bs2.length
        return false
    end

    for i in 1:length(bs1.data)
        if bs1.data[i] & ~bs2.data[i] != 0
            return false
        end
    end

    return true
end

function printBitSet(bs::CBitSet)
    buffer::String = ""
    for i in 0:bs.length-1
        if TestBit(bs, i)
            buffer *= "1"
        else
            buffer *= "0"
        end
    end

    println(buffer)
end

function makeRandomBitSets(n::Int, m::Int)
    sets = []

    for i in 1:m
        s = CBitSet(n)
        for j in 1:n
            if rand() < 0.5
                SetBit(s, j)
            end
        end
        push!(sets, s)
    end
    
    return sets
end

#=
function dict_perf(bitsets, n)
    x = Dict{CBitSet, Int}()
    
    sizehint!(x, n)
    for i = 1:n
        x[bitsets[i]] = i
    end
    return x
end

n = 10^7
bitsets = makeRandomBitSets(100, n)
@time dict_perf(bitsets, n)
@time dict_perf(bitsets, n)
=#

#= 
@time begin
    randomBitSets1 = makeRandomBitSets(100, 1000000)
    randomBitSets2 = makeRandomBitSets(100, 1000000)
end

@time begin
    for i in eachindex(randomBitSets1)
        isbitsubset(randomBitSets1[i], randomBitSets2[i])
    end
end

function makeRandomSets(n::Int, m::Int)
    sets = []

    for i in 1:m
        s = Set{Int8}()
        for j in 1:n
            if rand() < 0.5
                push!(s, j)
            end
        end
        push!(sets, s)
    end
    
    return sets
end

@time begin
    randomSets1 = makeRandomSets(100, 1000000)
    randomSets2 = makeRandomSets(100, 1000000)
end

@time begin
    for i in eachindex(randomSets1)
        ⊆(randomSets1[i], randomSets2[i])
    end
end

list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
bs = BitSet(list)
printBitSet(bs)
=#