from langchain_community.document_loaders import WebBaseLoader

class WebService:
    """
    Facade for fetching web page content using LangChain's WebBaseLoader.
    """

    @staticmethod
    def get_web_content(url: str) -> str:
        """
        Fetches and returns the main content of the web page at the given URL.

        Args:
            url (str): The URL of the web page to fetch.

        Returns:
            str: The extracted text content from the web page.
        """
        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            return ""
        # docs is a list of Document objects; extract their content
        return "\n\n".join(doc.page_content for doc in docs)