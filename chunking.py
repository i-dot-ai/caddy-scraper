import tiktoken
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

default_encoder = "cl100k_base"

encoding = tiktoken.get_encoding(default_encoder)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)


markdown_splitter_list = []


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def split_strings_by_logic(
    list_of_string: List[str],
    max_tokens: int,
    list_of_splitters: List[str],
    final_splitter: str,
) -> List[str]:
    """Splits a markdown string into a list of strings based on the logic of the markdown."""

    # Create an empty list to store the resulting strings
    result = []

    # Iterate over each markdown string in the list
    for markdown in list_of_string:
        # Get the number of tokens in the markdown string
        num_tokens = num_tokens_from_string(markdown, default_encoder)

        # If the number of tokens is less than or equal to max_tokens, add the markdown string to the result list
        if num_tokens <= max_tokens:
            result.append(markdown)
        else:
            # Split the markdown string using each splitter in the list
            split_strings = [
                splitter.split_text(markdown) for splitter in list_of_splitters
            ]

            # Flatten the list of split strings
            split_strings = [
                split_string for sublist in split_strings for split_string in sublist
            ]

            # Check if any of the resulting strings are shorter than or equal to max_tokens
            for split_string in split_strings:
                if num_tokens_from_string(split_string, default_encoder) <= max_tokens:
                    result.append(split_string)

            # If none of the splitters resulted in strings shorter than or equal to max_tokens, split using the final_splitter
            if len(result) == 0:
                split_strings = final_splitter.split_text(markdown)
                result.extend(split_strings)

    # Return the list of resulting strings
    return result


def add_token_length_to_document(document):
    """Adds the number of tokens in a document to the document."""

    document.metadata["token_count"] = num_tokens_from_string(
        document.page_content, default_encoder
    )

    return document


def split_documents_by_chunker(
    documents: List, max_tokens: int, final_splitter
) -> List[str]:
    # apply add_token_length_to_document to each document in the list

    document_list = [add_token_length_to_document(doc) for doc in documents]

    print("total document count:", len(document_list))

    list_of_too_long_docs = [
        doc for doc in document_list if doc.metadata["token_count"] > max_tokens
    ]

    list_of_short_docs = [
        doc for doc in document_list if doc.metadata["token_count"] <= max_tokens
    ]

    print("too long document count:", len(list_of_too_long_docs))

    if len(list_of_too_long_docs) > 0:
        shortened_documents = final_splitter.split_documents(list_of_too_long_docs)
        new_short_docs = [
            add_token_length_to_document(doc) for doc in shortened_documents
        ]
        list_of_short_docs.extend(new_short_docs)
        return list_of_short_docs

    return list_of_short_docs
