## Vim text editor

- i -> insert mode
- R -> replace mode
- 0 or o insert line below/above
- d -> delete motion
- dw -> delete word
- d$ -> delete to end of line
- d0 -> delete begninning of line
- cw -> change word
- x -> delete character
- s -> substitute character
- u -> undo
- y -> to copy means y{motion} yy y3j y$ yw
- p -> paste
- ~ -> flips the case of a character 
- v, V, ctrl+V -> types of selection
- ":help :w open help of w command
- Basic movement -> hjkl (left, down , up, right)
- movement on words -> w(next word), b(beginning of word), e (end of word)
- scroll -> Ctrl+u (up), Ctrl+d(down)
- file -> gg (beginnning of file) G (end of file)
- line number -> {number} G 
- editing paranthesis -> % jump between matching brackets
- dd -> delete the complete line
- / in command mode will activate search for you, say you want to search "datadir" simply type /datadir
- ?example -> search in the opposite direction (from the bottom of the file to the top)
- :set hlsearch -> to highlight search
- :set nohlsearch -> To turn off search highlighting
- %s/foo/bar/g -> it means replace foo with bar globally in file
- :sp split windows horizontally 
- :vsp split windows vertically
- :sp anotherfile.txt -> opens anotherfile.txt in new horizontal window
- ctrl-w will switch between windows
- ctrl-w j -> To switch to the split on the left
- ctrl-w k -> To switch to the split on the right
- ctrl-w + -> To increase the width of the current split
- ctrl-w - -> To decrease the width of the current split
- :tabnew new tab
- :q means quit
- :q! means quit without saving changes
- :only -> close all splits except the one you're currently in

