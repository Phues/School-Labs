import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class KMeans extends Configured implements Tool {	
	
	private static final int iterations = 10;
	
	public static void Iterate(Configuration conf) throws IOException {
    	//read initial centroid file in for first iteration and for each iteration after, use the output file as the new centroids
		FileSystem fs = FileSystem.get(conf);
		Path InitialCentroidFile = new Path("/input/initial_centroid.txt");
		Path NewCentroidFile = new Path("/output/part-r-00000");
		if(!(fs.exists(InitialCentroidFile) && fs.exists(NewCentroidFile)))
		{
			System.exit(1);
		}
		
		//rename the new centroid file to the old one because mapper uses the old name as input
		fs.delete(InitialCentroidFile, true);
		if(fs.rename(NewCentroidFile, InitialCentroidFile) == false)
		{
			System.exit(1);
		}
    	
	}
	
	
	public static class KMeansMapper extends Mapper<Object, Text, IntWritable, Text> {
		
	    public static List<Double[]> centroids = new ArrayList<>();

	    public void setup(Context context) throws IOException {
	    	   	
	    	try {
				Path[] cache = DistributedCache.getLocalCacheFiles(context.getConfiguration());
				if(cache == null || cache.length <= 0)
				{
					System.exit(1);
				}
				
				BufferedReader reader = new BufferedReader(new FileReader(cache[0].toString()));
				String line = "";
				
				while((line = reader.readLine()) != null) {
					String str1 = line.substring(line.indexOf("	")+1);
                	String[] str = str1.split(","); 
                	Double[] centroid = new Double[2];
      	  	      	
      		      	centroid[0] = Double.parseDouble(str[0]);
      		      	centroid[1] = Double.parseDouble(str[1]);
      		      	centroids.add(centroid);
				}
      		    reader.close();
	    	}
	    	catch(Exception e){
	    	}
	    }
	    

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
	        String[] xy = value.toString().split(",");
	        double x = Double.parseDouble(xy[0]);
	        double y = Double.parseDouble(xy[1]);
			double minDistance = Double.MAX_VALUE;
			int index = 0;
			  
			for (int j = 0; j < centroids.size(); j++) {
				 double cx = centroids.get(j)[0];
				 double cy = centroids.get(j)[1];
				
				 double distance = Math.sqrt(Math.pow(cx - x, 2) + Math.pow(cy - y, 2));
				 if (distance < minDistance) {
				 index = j;
				 minDistance = distance;
				 }
			}
			context.write(new IntWritable(index), value);
		}
	}
	
	
	
	public static class KMeansReducer extends Reducer<IntWritable, Text, Text, Text> {

	    protected void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

	        Double mx = 0d;
	        Double my = 0d;
	        int count = 0;

	        for (Text value: values) {
	            String[] xy = value.toString().split(",");
	            mx += Double.parseDouble(xy[0]);
	            my += Double.parseDouble(xy[1]);
	            count += 1;
	        }

	        mx = mx / count;
	        my = my / count;
	        String centroid = mx + "," + my;
	        String clusterId = "Cluster:" + key;
	        
	        context.write(new Text(clusterId), new Text(centroid));
	        
	    }

	}
	
	
	
	
	public int run(String[] args) throws Exception {
		
		Configuration conf = getConf();
		FileSystem fs = FileSystem.get(conf);
		Job job = new Job(conf);
		job.setJarByClass(KMeans.class);
		
		FileInputFormat.setInputPaths(job, "/input/data_points.txt");
		Path outDir = new Path("/output/newCentroid");
		fs.delete(outDir,true);
		FileOutputFormat.setOutputPath(job, outDir);
		 
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		
		job.setMapperClass(KMeansMapper.class);
		job.setReducerClass(KMeansReducer.class);
		
		job.setNumReduceTasks(1);
		
		job.setMapOutputKeyClass(IntWritable.class);
	    job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		 
		return job.waitForCompletion(true)?0:1;
	}
	
	
	public static void main(String[] args) throws Exception {
		
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		 
		Path dataFile = new Path("/input/initial_centroid.txt");
		DistributedCache.addCacheFile(dataFile.toUri(), conf);
 
		int iteration = 1;
		int success = 1;
		do {
			success ^= ToolRunner.run(conf, new KMeans(), args);
			iteration++;
		} while (success == 1 && iteration < iterations );
		 
		
		// final output
		
		Job job = new Job(conf);
		job.setJarByClass(KMeans.class);
		
		FileInputFormat.setInputPaths(job, "/input/data_points.txt");
		Path outDir = new Path("/output/final");
		fs.delete(outDir,true);
		FileOutputFormat.setOutputPath(job, outDir);
		 
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		
		job.setMapperClass(KMeansMapper.class);
		job.setReducerClass(KMeansReducer.class);
		
		job.setNumReduceTasks(1);
		
		job.setMapOutputKeyClass(IntWritable.class);
	    job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		 
		job.waitForCompletion(true);
		
	}
}
