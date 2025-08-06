from trainyolo.client import Client, Project

APIKEY = "8b0306827afeabf1b0f691ec0d456c6bbe9fb720"

# get client
client = Client(APIKEY)

# get project
name = 'Pancristal'
project = Project.get_by_name(client, name)

# export
export_path = './{}'.format('databases')
# export_format = 'yolov5' # or yolov8
export_format = 'yolov8' # or yolov8
project_path = project.pull(location=export_path, format=export_format)


select