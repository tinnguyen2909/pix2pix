import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class MyAlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It loads paired images from separate directories for domains A and B.
    Multiple directories can be specified using comma-separated paths.
    The number of directories in path_A and path_B must be equal.
    Images are matched within each corresponding directory pair based on filenames (without extensions).
    Images are loaded recursively from all specified directories.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        
        # Check if paths are provided
        if not hasattr(opt, 'path_A') or not hasattr(opt, 'path_B') or not opt.path_A or not opt.path_B:
            raise ValueError("Both path_A and path_B must be provided when using my_aligned dataset mode")
        
        # Parse comma-separated paths
        self.dirs_A = [os.path.join(p.strip(), opt.phase) if os.path.isdir(os.path.join(p.strip(), opt.phase))
                     else p.strip() for p in opt.path_A.split(',')]
        self.dirs_B = [os.path.join(p.strip(), opt.phase) if os.path.isdir(os.path.join(p.strip(), opt.phase))
                     else p.strip() for p in opt.path_B.split(',')]
        
        # Ensure equal number of directories
        if len(self.dirs_A) != len(self.dirs_B):
            raise ValueError(f"Number of directories in path_A ({len(self.dirs_A)}) and path_B ({len(self.dirs_B)}) must be equal")
        
        # Create paired path lists
        self.A_paths = []
        self.B_paths = []
        
        # Process each directory pair
        total_pairs = 0
        for dir_A, dir_B in zip(self.dirs_A, self.dirs_B):
            # Check if directories exist
            if not os.path.isdir(dir_A):
                print(f"Warning: Directory not found: {dir_A}")
                continue
            if not os.path.isdir(dir_B):
                print(f"Warning: Directory not found: {dir_B}")
                continue
            
            # Get all image paths from directories
            A_paths = sorted(make_dataset(dir_A, opt.max_dataset_size, recursive=True))
            B_paths = sorted(make_dataset(dir_B, opt.max_dataset_size, recursive=True))
            
            if not A_paths:
                print(f"Warning: No images found in {dir_A}")
                continue
            if not B_paths:
                print(f"Warning: No images found in {dir_B}")
                continue
            
            print(f"Found {len(A_paths)} images in {dir_A} and {len(B_paths)} images in {dir_B}")
            
            # Create dictionaries mapping filename (without extension) to full path
            A_path_dict = {}
            for path in A_paths:
                basename = os.path.splitext(os.path.basename(path))[0]
                A_path_dict[basename] = path
                
            B_path_dict = {}
            for path in B_paths:
                basename = os.path.splitext(os.path.basename(path))[0]
                B_path_dict[basename] = path
                
            # Find common filenames between A and B
            common_names = set(A_path_dict.keys()) & set(B_path_dict.keys())
            
            # Add paired paths to our lists
            dir_pairs = 0
            for name in sorted(common_names):
                self.A_paths.append(A_path_dict[name])
                self.B_paths.append(B_path_dict[name])
                dir_pairs += 1
                
            total_pairs += dir_pairs
            print(f"Created {dir_pairs} matching A-B image pairs from directories {dir_A} and {dir_B}")
        
        if not self.A_paths:
            raise ValueError("No matching image pairs found across any directory pairs!")
            
        print(f"Total: Created {total_pairs} matching A-B image pairs from all directory pairs")
            
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths
        """
        # Load images from separate paths
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        # Apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--path_A', type=str, required=True, help='path to directory containing domain A images (comma-separated for multiple directories)')
        parser.add_argument('--path_B', type=str, required=True, help='path to directory containing domain B images (comma-separated for multiple directories)')
        return parser
