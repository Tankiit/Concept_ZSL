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
    subsets = Array{Set{Int8}}()
    for feature in feature_set
        push!(subsets, Set{Int8}([feature]))
    end
    return subsets
end

function extend_feature_bank(feature_bank)
    for (class_, feature_list) in feature_bank
        new_feature_list = [feature_list[1]]
        source_feature_sets = [feature_list2[1] for (class2, feature_list2) in feature_bank if class_ != class2]

        subsets = get_direct_subsets(feature_list[1])
        while length(subsets) > 0
            new_subsets = Array{Set{Int8}}()
            for subset in subsets
                to_add = true
                for source_feature_set in source_feature_sets
                    if subset.issubset(source_feature_set)
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
        end

        feature_bank[class_] = new_feature_list
    end

    return feature_bank
end

feature_bank = extend_feature_bank(feature_bank)
