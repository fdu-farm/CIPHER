const copyButton = document.querySelector('#copy-citation');
const citation = document.querySelector('#bibtex');
const copyStatus = document.querySelector('#copy-status');

copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyStatus.textContent = 'Citation copied to clipboard.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citation);
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
