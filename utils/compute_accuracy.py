from __future__ import print_function, absolute_import

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    """
        tensor.topk return a tupla: 
            first, the k top predictions values, 
            second, the indices of those k values, id., the classes, that are our variable pred
    """
    pred = pred.t() # Transpose to finally do each raw vs target
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # Comparison, it is a boolean matrix at the end

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0) # How many predictions are correct per topk
        res.append(correct_k.mul_(100.0 / batch_size)) # Calculate the final accuracy per topk
    return res