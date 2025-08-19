"""
Test Questions for RAGAs Evaluation
Location: src/contextual_rag/evaluation/testing/test_questions.py
"""

# Primary test questions for RAG evaluation (from your reference)
TEST_QUESTIONS = [
    "How do the \"Delivery Terms\" and \"Payment Terms\" relate to a \"Purchase Order\" within the procurement process described in this document?",
    
    "What are the specific sub-controls listed under \"T5.2.3 USER SECURITY CREDENTIALS MANAGEMENT\"?"
]

# Full question set (can be used for comprehensive evaluation)
FULL_TEST_QUESTIONS = [
    "How do the \"Delivery Terms\" and \"Payment Terms\" relate to a \"Purchase Order\" within the procurement process described in this document?",
    
    "Infer the hierarchical relationship between Law No. (6) of 2016 and Decision No. (10) of 2020, based on their descriptions in the document.",
    
    "Based on the penalties section, what are the different levels of disciplinary actions for various infringements, and how do they relate to preserving pension or bonus rights?",
    
    "What are the specific sub-controls listed under \"T5.2.3 USER SECURITY CREDENTIALS MANAGEMENT\"?"
]

# Additional questions for comprehensive evaluation
EXTENDED_TEST_QUESTIONS = [
    "What are the key procurement standards mentioned in Abu Dhabi documents?",
    "What security requirements are mentioned for supplier agreements?",
    "What HR policies should employees be aware of?",
    "What is the procurement process workflow according to the standards?",
    "What are the confidentiality requirements for suppliers?",
    "What are the information security guidelines for data protection?",
    "How should procurement practitioners manage vendor relationships?",
    "What are the compliance requirements for government procurement?",
    "What are the procurement approval requirements?",
    "What are the information security policies for data handling?",
    "What are the HR leave policies and procedures?",
    "How do procurement evaluation criteria work?",
    "What are the vendor selection requirements?",
    "What security controls are required for information systems?"
]

def get_test_questions(extended=False, full_set=False):
    """
    Get test questions for evaluation
    
    Args:
        extended (bool): Whether to include extended question set
        full_set (bool): Whether to use all 4 original questions
    
    Returns:
        list: List of test questions
    """
    if full_set:
        base_questions = FULL_TEST_QUESTIONS
    else:
        base_questions = TEST_QUESTIONS  # 2 questions for quick testing
        
    if extended:
        return base_questions + EXTENDED_TEST_QUESTIONS
    return base_questions

def get_question_categories():
    """
    Categorize questions by domain for targeted evaluation
    
    Returns:
        dict: Questions categorized by domain
    """
    categories = {
        "procurement_process": [
            "How do the \"Delivery Terms\" and \"Payment Terms\" relate to a \"Purchase Order\" within the procurement process described in this document?",
            "What are the key procurement standards mentioned in Abu Dhabi documents?",
            "What is the procurement process workflow according to the standards?",
            "What are the procurement approval requirements?",
            "How do procurement evaluation criteria work?",
            "What are the vendor selection requirements?"
        ],
        "legal_hierarchy": [
            "Infer the hierarchical relationship between Law No. (6) of 2016 and Decision No. (10) of 2020, based on their descriptions in the document."
        ],
        "hr_policies": [
            "Based on the penalties section, what are the different levels of disciplinary actions for various infringements, and how do they relate to preserving pension or bonus rights?",
            "What HR policies should employees be aware of?",
            "What are the HR leave policies and procedures?"
        ],
        "security_controls": [
            "What are the specific sub-controls listed under \"T5.2.3 USER SECURITY CREDENTIALS MANAGEMENT\"?",
            "What are the information security guidelines for data protection?",
            "What security requirements are mentioned for supplier agreements?",
            "What are the information security policies for data handling?",
            "What security controls are required for information systems?"
        ],
        "compliance": [
            "What are the compliance requirements for government procurement?",
            "What are the confidentiality requirements for suppliers?",
            "How should procurement practitioners manage vendor relationships?"
        ]
    }
    return categories

def get_questions_by_category(category: str):
    """
    Get questions for a specific category
    
    Args:
        category (str): Category name
    
    Returns:
        list: Questions in the specified category
    """
    categories = get_question_categories()
    return categories.get(category, [])

def get_evaluation_dataset_info():
    """
    Get information about the test dataset
    
    Returns:
        dict: Dataset information
    """
    return {
        "total_questions": {
            "basic": len(TEST_QUESTIONS),
            "full": len(FULL_TEST_QUESTIONS),
            "extended": len(TEST_QUESTIONS + EXTENDED_TEST_QUESTIONS)
        },
        "categories": list(get_question_categories().keys()),
        "questions_per_category": {
            cat: len(questions) 
            for cat, questions in get_question_categories().items()
        },
        "recommended_for_testing": "basic",
        "recommended_for_comprehensive": "full"
    }

if __name__ == "__main__":
    print("📋 Test Questions for RAGAs Evaluation")
    print("=" * 60)
    
    print("🔥 Basic Test Questions (Recommended):")
    questions = get_test_questions()
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    
    print(f"\nTotal basic questions: {len(questions)}")
    
    print("\n🚀 Full Question Set:")
    full_questions = get_test_questions(full_set=True)
    for i, question in enumerate(full_questions, 1):
        print(f"{i}. {question[:80]}...")
    
    print(f"Full set total: {len(full_questions)}")
    
    print("\n📊 Questions by Category:")
    categories = get_question_categories()
    for category, cat_questions in categories.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for question in cat_questions:
            print(f"  - {question[:80]}...")
    
    print("\n📋 Dataset Information:")
    info = get_evaluation_dataset_info()
    for key, value in info.items():
        print(f"{key}: {value}")