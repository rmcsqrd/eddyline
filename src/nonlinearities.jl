function relu(x)
    return max(0, x)
end

function softmax(x̄)
    exp = [ℯ^x for x in x̄]
    return exp ./ sum(exp)
end
