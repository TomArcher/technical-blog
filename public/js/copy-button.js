document.addEventListener('DOMContentLoaded', function() {
    const codeBlocks = document.querySelectorAll('div.highlight > pre');
    
    codeBlocks.forEach(function(pre) {
        if (pre.querySelector('.copy-button')) return;
        
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        
        button.addEventListener('click', function() {
            const codeElement = pre.querySelector('code');
            if (!codeElement) return;

            // Get the innerText
            let text = codeElement.innerText;

            // Replace ONLY double blank lines (not all double newlines)
            // This regex keeps single blank lines but removes double ones
            text = text.replace(/\n\n\n+/g, '\n\n');  // Replace 3+ newlines with 2
            text = text.replace(/\n\n/g, '\n');        // Then replace 2 newlines with 1

            // But we want to preserve intentional blank lines in code
            // So let's be more selective - only remove if EVERY line break is doubled
            // First, let's check if this looks like everything is double-spaced
            const lines = codeElement.innerText.split('\n');
            let hasPattern = true;

            // Check if every other line is blank (sign of double spacing)
            for (let i = 1; i < Math.min(lines.length, 10); i += 2) {
                if (lines[i] !== '') {
                    hasPattern = false;
                    break;
                }
            }

            if (hasPattern) {
                // It's double-spaced, remove every other line
                text = lines.filter((_, index) => index % 2 === 0).join('\n');
            } else {
                // Just remove excessive blank lines but keep single ones
                text = text.replace(/\n\n\n+/g, '\n\n');
            }

            // Trim the result
            text = text.trim();
            
            navigator.clipboard.writeText(text).then(function() {
                button.textContent = 'Copied!';
                setTimeout(function() {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
        
        pre.style.position = 'relative';
        pre.appendChild(button);
    });
});