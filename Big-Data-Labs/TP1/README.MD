# Big Data Lab

## Big Data Lab

To set up the Big Data Lab environment, follow these steps:

1. Download the docker image from Docker Hub:
    
    ```
    docker pull liliasfaxi/spark-hadoop:hv-2.7.2
    
    ```
    
2. Create the three containers using the downloaded image. Perform the following sub-steps:
    1. Create a network to connect the three containers:
        
        ```
        docker network create --driver=bridge hadoop
        
        ```
        
    2. Create and launch the three containers. Use the `p` flag to map the host machine ports to the container ports:
        
        ```
        docker run -itd --net=hadoop -p 50070:50070 -p 8088:8088 -p 8080:8080^
        --name hadoop-master --hostname hadoop-master^
        liliasfaxi/spark-hadoop:hv-2.7.2
        
        docker run -itd -p 8040:8042 --net=hadoop^
        --name hadoop-slave1 --hostname hadoop-slave1^
        liliasfaxi/spark-hadoop:hv-2.7.2
        
        docker run -itd -p 8041:8042 --net=hadoop^
        --name hadoop-slave2 --hostname hadoop-slave2^
        liliasfaxi/spark-hadoop:hv-2.7.2
        
        ```
        
3. Enter the master container to start using it:
    
    ```
    docker exec -it hadoop-master bash
    
    ```
    
    To start the Hadoop services, run the following command in the shell of the master container:
    
    ```bash
    ./start-hadoop.sh
    
    ```
    
    1. Create a directory in HDFS called "input" by running the following command:
        
        ```
        hadoop fs -mkdir -p input
        
        ```
        
        We will use the file "purchases.txt" as the input for the MapReduce processing.
        
        - Load the "purchases" file into the "input" directory you created by running the following command:
            
            ```
            hadoop fs -put purchases.txt input
            
            ```
            
            Create a Maven project with the required files:
            
            1. Open Intelij IDE (using JDK 8)
            
            Install the required dependencies, add the following code to `pom.xml` file:
            
            ```xml
            <dependencies>
                <dependency>
                    <groupId>org.apache.hadoop</groupId>
                    <artifactId>hadoop-common</artifactId>
                    <version>2.7.2</version>
                </dependency>
                <dependency>
                    <groupId>org.apache.hadoop</groupId>
                    <artifactId>hadoop-mapreduce-client-core</artifactId>
                    <version>2.7.2</version>
                </dependency>
                <dependency>
                    <groupId>org.apache.hadoop</groupId>
                    <artifactId>hadoop-hdfs</artifactId>
                    <version>2.7.2</version>
                </dependency>
                <dependency>
                    <groupId>org.apache.hadoop</groupId>
                    <artifactId>hadoop-mapreduce-client-common</artifactId>
                    <version>2.7.2</version>
                </dependency>
            </dependencies>
            
            ```
            
            1. In the project structure, navigate to the source directory (`src/main/java`).
                - Create a new package called `tp1`.
                - Inside the `tp1` package, create the following Java classes:
                    - `TokenizerMapper.java`
                    - `IntSumReducer.java`
                    - `WordCount.java`
            2. TokenizeMapper.java:
            
            ```java
            package tp1;
            
            import org.apache.hadoop.mapreduce.Mapper;
            import org.apache.hadoop.io.Text;
            import org.apache.hadoop.io.IntWritable;
            import java.util.StringTokenizer;
            import java.io.IOException;
            
            public class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
            
                private static final IntWritable ONE = new IntWritable(1);
            
                protected void map(Object offset, Text value, Context context)
                    throws IOException, InterruptedException
                {
                    StringTokenizer tok = new StringTokenizer(value.toString(), " ");
            
                    while (tok.hasMoreTokens())
                    {
                        Text word = new Text(tok.nextToken());
            
                        context.write(word, ONE);
                    }
                }
            }
            
            ```
            
            IntSumReducer.java
            
            ```java
            package tp1;
            
            import org.apache.hadoop.io.Text;
            import org.apache.hadoop.io.IntWritable;
            import org.apache.hadoop.mapreduce.Reducer;
            import java.util.Iterator;
            import java.io.IOException;
            
            public class IntSumReducer extends Reducer<Text, IntWritable, Text, Text> {
            
                public void reduce(Text key, Iterable<IntWritable> values, Context context)
                    throws IOException, InterruptedException
                {
                    Iterator<IntWritable> i= values.iterator();
                    int count = 0;
                    while(i.hasNext())
                        count += i.next().get();
            
                    context.write(key, new Text(count+" occurences"));
                }
            
            }
            
            ```
            
            WordCount.java
            
            ```java
            package tp1;
            
            import org.apache.hadoop.fs.Path;
            import org.apache.hadoop.mapreduce.Job;
            import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
            import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
            import org.apache.hadoop.conf.Configuration;
            import org.apache.hadoop.util.GenericOptionsParser;
            import org.apache.hadoop.io.Text;
            import org.apache.hadoop.io.IntWritable;
            
            public class WordCount {
                public static void main(String[] args) throws Exception
                {
                    Configuration conf = new Configuration();
            
                    String[] ourArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
            
                    Job job=Job.getInstance(conf, "Compteur de mots v1.0");
            
                    job.setJarByClass(WordCount.class);
                    job.setMapperClass(TokenizerMapper.class);
                    job.setReducerClass(IntSumReducer.class);
            
                    job.setOutputKeyClass(Text.class);
                    job.setOutputValueClass(IntWritable.class);
            
                    FileInputFormat.addInputPath(job, new Path(ourArgs[0]));
                    FileOutputFormat.setOutputPath(job, new Path(ourArgs[1]));
            
                    if(job.waitForCompletion(true))
                        System.exit(0);
                    System.exit(-1);
                }
            }
            
            ```
            
        - Copy the created jar file into the master container:
            - Open the terminal
            - Execute the following command to copy the jar file:
                
                ```
                docker cp target/wordcount-1.jar hadoop-master:/root/wordcount-1.jar
                
                ```
                
    2. Return to the shell of the master container and launch the MapReduce job with the following command:
        
        ```
        hadoop jar wordcount-1.jar tp1.WordCount input output
        
        ```
Result:
![result](https://github.com/Phues/Big-Data-Labs/blob/main/TP1/image.png)

## Python Version 
 ```
 hadoop jar hadoop-streaming-2.7.3.jar  -file ./mapper.py -mapper "python3 mapper.py" -file ./reducer.py -reducer "python3 reducer.py" -input input/purchases.txt -output pyout
        
 ```
