from adaptive_qa.answering.simulated import extract_answer_span, question_type


def test_question_types():
    assert question_type("Who invented the lightbulb?") == "who"
    assert question_type("When was it built?") == "when"
    assert question_type("Where is the museum?") == "where"
    assert question_type("How many people attended?") == "quantity"
    assert question_type("What is the largest lake?") == "what"
    assert question_type("Are both A and B French?") == "comparison"


def test_extract_year():
    span = extract_answer_span("When was Pixar founded?", "Pixar Animation Studios was founded in 1986 by Ed Catmull.")
    assert span == "1986"


def test_extract_person():
    span = extract_answer_span("Who founded Pixar?", "Pixar Animation Studios was founded by Ed Catmull.")
    assert span == "Ed Catmull"


def test_extract_from_capitalized_span():
    span = extract_answer_span("Where is the Louvre?", "The Louvre is located in Paris, France.")
    assert span is not None and "Paris" in span


def test_extraction_fails_gracefully():
    assert extract_answer_span("When was it founded?", "It was founded then, after a long delay.") is None
    assert extract_answer_span("Are both X and Y French?", "Some unrelated sentence text.") is None
