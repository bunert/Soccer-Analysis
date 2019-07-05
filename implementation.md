## Data structure:

### Dictionaries (different camera seperated)

#### data_dict
Data structure (dictionary) with the complete SoccerVideo class data in it.
```python
data_dict = {0:db_K1, 1:db_K8, 2:db_K9, 3:db_Left, 4:db_Right}
# access: data_dict[0] to get access to a specific class object
```

#### keypoint_dict
A Dictionary with an entry for each camera, one entry is an unordered list of all the persons where there are keypoints for this specific frame. So one entry in this list is a matrix where one row is one keypoints (x,y Screen coordinates and precision).

Important: this just gets the informations from the SoccerVideo classes for one specified frame.

```python
keypoint_dict = {0:db_K1_poses, 1:db_K8_poses, 2:db_K9_poses, 3:db_Left_poses, 4:db_Right_poses}

# one data point:
[x, y, prec]
```

#### projected_players_2d_dict
A Dictionary with an entry for each camera.
One camera entry is a list of all players from players_3d projected to the camera screen.

#### players_2d_dict
A Dictionary with an entry for each camera.
One camera entry is a list of tuples of all players where openpose found a pose. The first entry in the tuple is the player number (0-10 Danmark, 11-21 Swiss), the second entry are the keypoints with precision from openpose.
```python
# tuple:
(player_number, 18x3 Matrix)
```

### Lists

#### players_3d
List with all players (0-10 Danmark, 11-21 Swiss), each element of the list is a matrix of the player. For initialization those 3D Points are hardcoded with the plane coordinates from the csv file.
```
# one element of the list:
[x, y, z]
[x, y, z]
[x, y, z]
```
