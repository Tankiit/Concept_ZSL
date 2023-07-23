using DelimitedFiles
predicate_matrix = readdlm("datasets/Animals_with_Attributes2/predicate-matrix-binary.txt", Int8)
println("Predicate matrix size: ", size(predicate_matrix))

features = readdlm("datasets/Animals_with_Attributes2/predicates.txt", String)[:, 2]
classes = readdlm("datasets/Animals_with_Attributes2/classes.txt", String)[:, 2]
println("Features size: ", size(features))
println("Classes size: ", size(classes))

feature_bank = Dict{String, Array{Set{Int8}}}()
for i in 1:size(predicate_matrix, 1)
    class_ = classes[i]
    feature_set = Set{Int8}()
    for j in 1:size(predicate_matrix, 2)
        if predicate_matrix[i, j] == 1
            push!(feature_set, j-1)
        end
    end

    if haskey(feature_bank, class_)
        push!(feature_bank[class_], feature_set)
    else
        feature_bank[class_] = [feature_set]
    end
end

function get_direct_subsets(feature_set::Set{Int8})
    subsets = []
    for feature in feature_set
        new_set = deepcopy(feature_set)
        delete!(new_set, feature)
        push!(subsets, new_set)
    end
    return subsets
end

using ProgressBars

function extend_feature_bank(feature_bank::Dict, max_new_features=100)
    for i in ProgressBar(1:length(classes))
        class_ = classes[i]
        feature_list = feature_bank[class_]

        new_feature_list = [feature_list[1]]
        source_feature_sets = [feature_list2[1] for (class2, feature_list2) in feature_bank if class_ != class2]

        subsets = get_direct_subsets(feature_list[1])
        while length(subsets) > 0
            new_subsets = []
            for subset in subsets
                to_add = true
                for source_feature_set in source_feature_sets
                    if issubset(subset, source_feature_set)
                        to_add = false
                        break
                    end
                end
                if to_add && !(subset in new_feature_list)
                    push!(new_feature_list, subset)
                    new_subsets = vcat(new_subsets, get_direct_subsets(subset))
                end
            end
            subsets = new_subsets

            if length(new_feature_list) >= max_new_features
                break
            end
        end

        feature_bank[class_] = new_feature_list
    end

    return feature_bank
end

println("Extending feature bank...")
feature_bank = extend_feature_bank(feature_bank)

# Save feature bank to json
println("Saving data...")
using JSON

data = Dict{String, Any}()
data["features"] = features
data["classes"] = classes
data["feature_bank"] = feature_bank

open("datasets/Animals_with_Attributes2/data.json", "w") do f
    JSON.print(f, data)
end