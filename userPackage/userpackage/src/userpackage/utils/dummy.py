
def dummy_prog(n: int) -> None:
    """Prints numbers from 1 to n
    
    Args:
        n (int): The upper limit of the range to print numbers (inclusive)
        
    Returns:
        None
        
    Example:
        dummy_prog(8) will print:
        1 2 3 4 5 6 7 8        
    
    """
    for i in range(1, n + 1):
        print(i, end=' ')
    print()
