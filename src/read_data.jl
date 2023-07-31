include("CBitSet.jl")
using DelimitedFiles
classes = readdlm("datasets/Animals_with_Attributes2/classes.txt", String)[:, 2]

bitsets = []

# Read from file
for class_ in classes
    file = open("datasets/Animals_with_Attributes2/feature_sets/$class_.bin", "r")

    # First 64 bits are the length of the bitset
    bitset_length = read(file, Int64)

    # Read the bitsets
    while !eof(file)
        bitset = CBitSet(bitset_length)
        for i in 1:length(bitset.data)
            bitset.data[i] = read(file, UInt64)
        end
        push!(bitsets, bitset)
    end

    close(file)
end
