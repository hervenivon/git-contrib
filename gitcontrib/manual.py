

def getOptimizedIndividual(expected_calendar,
                           actual_calendar,
                           shape,
                           flatshape,
                           nbclass,
                           verbose=False):
    manual_new = expected_calendar * 10 - actual_calendar
    manual_new[manual_new < 0] = 0

    return manual_new
