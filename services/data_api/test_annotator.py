from annotators import UnetELMDataAnnotator

annotator = UnetELMDataAnnotator()
peaks = annotator.get_annotations(30421)
