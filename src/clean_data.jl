using DelimitedFiles
predicate_matrix = readdlm("datasets/Animals_with_Attributes2/predicate-matrix-binary.txt", Int8)
println("Predicate matrix size: ", size(predicate_matrix))

features = readdlm("datasets/Animals_with_Attributes2/predicates.txt", String)[:, 2]
classes = readdlm("datasets/Animals_with_Attributes2/classes.txt", String)[:, 2]
println("Features size: ", size(features))
println("Classes size: ", size(classes))

include("CBitSet.jl")
feature_bank = Dict{String, CBitSet}()

bitset_length = 85

for i in 1:size(predicate_matrix, 1)
    class_ = classes[i]
    feature_set = CBitSet(bitset_length)
    for j in 1:size(predicate_matrix, 2)
        if predicate_matrix[i, j] == 1
            SetBit(feature_set, j-1)
        end
    end

    feature_bank[class_] = feature_set
end

function get_direct_subsets(feature_set::CBitSet)::Array{CBitSet}
    subsets = []
    sizehint!(subsets, length(feature_set))
    for feature in feature_set
        new_set = copy(feature_set)
        SetBit(new_set, feature, false)
        push!(subsets, new_set)
    end
    return subsets
end

using ProgressBars

#=

Start generating all subsets from the bottom-up. Each step, check if subset of the others. If not, add all intermediary subsets between it and the base one.
Generate the next layer of subsets, and pass the subsets of layer n-1 that where added, to prevent checking already validated subsets.

OR

Find the set of smallest unique generating subsets (subsest for which, if you make all intermediary sets, you get all valid unique subsets)

=#
using Dates
function extend_feature_bank(feature_bank::Dict, max_new_features::Integer=100)::Dict
    visited = Dict{CBitSet, Bool}()
    feature_set_files = Dict{String, String}()
    sizehint!(feature_set_files, length(classes))
    sizehint!(visited, 10^8)

    for i in ProgressBar(1:length(classes))
        class_ = classes[i]
        base_features = feature_bank[class_]

        source_feature_sets = [base_features2 for (class2, base_features2) in feature_bank if class_ != class2]

        subsets = get_direct_subsets(base_features)
        added_features = 0
        filename = "datasets/Animals_with_Attributes2/feature_sets/$class_.bin"
        feature_set_files[class_] = filename
        file = open(filename, "a")
        write(file, bitset_length)
        stop_going = false
        while length(subsets) > 0 && !stop_going
            new_subsets = []
            for subset in subsets
                if haskey(visited, subset)
                    continue
                end
                visited[subset] = true
                to_add = true
                for source_feature_set in source_feature_sets
                    if isbitsubset(subset, source_feature_set)
                        to_add = false
                        break
                    end
                end
                
                if to_add
                    for uint in subset.data
                        write(file, uint)
                    end
                    added_features += 1
                    if added_features >= max_new_features
                        stop_going = true
                        break
                    end
                    direct_subsets = get_direct_subsets(subset)

                    new_subsets = vcat(new_subsets, direct_subsets)
                end
                
            end
            subsets = new_subsets
        end

        close(file)
    end

    return feature_set_files
end

println("Extending feature bank...")
feature_set_files = extend_feature_bank(feature_bank, 10000)

# Save feature bank to json
println("Saving data...")
using JSON

data = Dict{String, Any}()
data["features"] = features
data["classes"] = classes
data["feature_set_files"] = feature_set_files

open("datasets/Animals_with_Attributes2/data.json", "w") do f
    JSON.print(f, data)
end