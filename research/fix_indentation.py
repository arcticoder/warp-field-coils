#!/usr/bin/env python3
"""
Fix indentation for helper methods in run_unified_pipeline.py
"""

import re

def fix_indentation():
    with open('run_unified_pipeline.py', 'r') as f:
        content = f.read()
    
    # Find the start of helper methods section
    helper_start = content.find('def _compute_signal_quality(self, phase_shifts: np.ndarray) -> Dict:')
    if helper_start == -1:
        print("Helper methods section not found")
        return
    
    # Split content
    before_helpers = content[:helper_start]
    helpers_section = content[helper_start:]
    
    # Fix indentation by adding 4 spaces to each line that needs it
    lines = helpers_section.split('\n')
    fixed_lines = []
    
    in_method = False
    for line in lines:
        if line.strip().startswith('def _'):
            # Method definition - ensure it starts with 4 spaces
            fixed_lines.append('    ' + line.strip())
            in_method = True
        elif line.strip() == '' or line.strip().startswith('#'):
            # Empty lines and comments
            fixed_lines.append(line)
        elif line.strip().startswith('"""') or line.strip().startswith('"""'):
            # Docstrings
            if not line.startswith('    '):
                fixed_lines.append('        ' + line.strip())
            else:
                fixed_lines.append(line)
        elif in_method and line.strip():
            # Method content - ensure proper indentation
            if not line.startswith('    '):
                if line.strip().startswith('return') or line.strip().startswith('if') or line.strip().startswith('try'):
                    fixed_lines.append('        ' + line.strip())
                else:
                    fixed_lines.append('        ' + line.strip())
            else:
                # Already indented, check if it needs more
                stripped = line.lstrip()
                current_indent = len(line) - len(stripped)
                if current_indent < 8:  # Method content should be at least 8 spaces
                    fixed_lines.append('        ' + stripped)
                else:
                    fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Reconstruct content
    fixed_content = before_helpers + '\n'.join(fixed_lines)
    
    with open('run_unified_pipeline.py', 'w') as f:
        f.write(fixed_content)
    
    print("Indentation fixed!")

if __name__ == '__main__':
    fix_indentation()
