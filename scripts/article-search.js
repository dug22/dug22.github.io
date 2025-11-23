document.addEventListener('DOMContentLoaded', function () {
        const searchInput = document.getElementById('article-search-input');
        const articlesContainer = document.getElementById('articles-container');
        const articles = articlesContainer.querySelectorAll('article');

        const noResultsMessage = document.createElement('div');
        noResultsMessage.className = 'no-results-message';
        noResultsMessage.innerHTML = `
    <p>No articles found matching your search.</p>
    <p>
      Try different keywords or clear the search.
    </p>
  `;
        noResultsMessage.style.display = 'none';

        articlesContainer.after(noResultsMessage);

        function performSearch() {
            const query = searchInput.value.trim().toLowerCase();
            let visibleCount = 0;

            articles.forEach(article => {
                const title = (article.getAttribute('data-title') || '').toLowerCase();
                const content = (article.getAttribute('data-content') || '').toLowerCase();
                const searchableText = title + ' ' + content;

                if (query === '' || searchableText.includes(query)) {
                    article.style.display = '';
                    visibleCount++;
                } else {
                    article.style.display = 'none';
                }
            });

            noResultsMessage.style.display = (visibleCount === 0 && query !== '') ? 'block' : 'none';
        }

        searchInput.addEventListener('input', performSearch);
        searchInput.addEventListener('search', performSearch);
    });
