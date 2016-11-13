__all__ = ['parseFile']

def parseFile(path):
    """
    Take a path to a textfile of newline-delimited
    key/value pairs and return a dictionary of strings
    mapped to strings or strings mapped to lists:

    examples:

        key: value    ->    {key: value}
        key: value1, value2 -> {key: [value1, value2]}
    """
    returnDict = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, colon, value = line.partition(':')
            if not colon:
                continue
        
            key = key.strip()
            value = [item.strip() for item in value.split(',')]
            if len(value) == 1:
                value = value[0]

            returnDict[key] = value

    return returnDict
