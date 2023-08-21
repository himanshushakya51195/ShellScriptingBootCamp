#!/bin/bash

<< COMMENT 
	
	creating folders by given list

COMMENT

name_list=(
"folder1"
"folder2"
"folder3"
"folder4"
)

create_folders ()
{
	for folder_name in "${name_list[@]}"; do
		mkdir -p "$folder_name"
	done			
}

: ' call
	 the
	 	function'

create_folders


