remove unused code
fix build failure
expanded visibility of inner class members in baseimagedecoder
merge branch master of github com mrniko redisson
give debugcontext description a tostring clear metric values in debugcontext close
hotfix fix ioutils closeall exceptions aggregation
small typo on illegalargumentexception
add valueproxy interface to fixedvalueanchornode
linking file input click test to a bug report
jfr for svm must read system properties in the same way as svm code
stronger error for vtable out of range
unsafearraycopynode location identity must be any_location as it lowers to snippets containing nodes with any_location
update shellsort java
rsv active panel settings save fix
jsonreader add type reference support
fix checkstyle violation remove unused import
fix format gate failure
test increased logging
move rowcounttoken to rewrite package
use new structure of distinctqueryresult
hotfix docs fix typo in configgroups
remove todo for timeout feature on info cache
add missing since to impersonate
generalized signature of setinputs in genericdatasink
remove race in testdisableidletimeoutinpool
reduce number of unused parameters
truffle canonicalize inlined invoke usages during partial evaluation
fix a missing closing tag in requestmanager javadoc
call simpler interpolation method
use full explode until return loop explosion for hybrid bit set matcher
add jacoco exclude to graphprinterdumphandler and benchmarkcounters problems with checkgraalinvariants
use collection string
improved layout algorithm to put an emphasis on cfg edges
merge branch master into feature list languages
update license year
hotfix metrics add missing override annotations
clean unused method
remove invalid import
for now keep both java and android module index
make sure groupname is calculated
revert back applying mapping changes on index changes
introduce trap in deadendop
fix r noclassdeffound bug
remove unnecessary conditional
add project name and result for run api
remove unuse import
only remove the first occurrence
fix author tag
fix spelling error
useless import deleted
href target so it opens from javadoc frame
test testnodenotreachablefrommaster had a typo when choosing a non master node
made constructor visible
fix to avoid emitting a range switch for non integer keys fix for reading a value from a metadata object
update after review
polish javadoc for injectionmetadata forelements
javadocs add getoverobject
healthreport add missing override annotation
add channelwritabilitychanged hook but not implement yet
failing test case
fix the operation name of spring mvc span is incorrect
use illegal equals value in valueutil isillegal
more minor cleanup
add getalldaos to daosession
minor javadoc polish
add io shardingsphere core keygen uuidkeygenerator
relaxed map string object to map string for method parameter
allow skipping if you hit a network error during pin restore
createprojectfromxml is not supposed to be called for a project which already exists but failed to enforce this
operatordoonrequest parentsubscriber should be static class
add missing override annotations
polish configurablepropertyresolver javadoc
do not allow a batch size greater than the ring buffer size
legal stuff should this be done by the maven license plugin
make threadlocalhandles nullhandle uninterruptible
hotfix remove deprecated calls
fix checkstyle issues
make engine tracecompilationdetails an expert option too for consistency
log connection errors instead of throwing fatal exceptions
add fallback_missing counter to codahale metrics publisher
simplify defaultbootstrapcontext tostring
fix rebase gone wrong leaving duplicate methods
fix settings for tribeunittests
fix warning in rectangle documentation
remove ineffective assert statement
wires is kaput stop referring to it now we only look for geckodriver
add default x spacing half width for scatter chart as well
if we already have an idset use it
don t allow checkbox image to be squashed
change use of abstractannotationvaluevisitor7 to abstractannotationvaluevisitor8
decodetext implementation is reformatted
fix local tee instruction in binaryreader restore block state checking
pg index drop fix index cache fix
druiddatasource init remove maxidle check
fixes after rebase
update core src main java hudson slaves slavecomputer java co authored by jesse glick jglick cloudbees com
more specific assertion on fireandforget error
isauthorizedschema final string schema
hotfix javadoc remove wrong javadoc from subtaskmetricshandler
use equals rather than for string equality
fix a whitespace and spelling issue that was causing the test case to fail
remove unused import
fix incorrect return value in codahale metrics publisher of threadpool config
test travis ut works fine
phase contracts disable contract check for frame state assignment temporarily
add the number of nio loops created by vert x
modify the move thread
remove request for permission that doesn t exist o
remove unused methods
fix flaky test in issue1769
add comment in front of main method
add a new section in lombok builder article minor fixes
make isdefaultconstant access any constant
hotfix runtime correct some javadocs for restartindividualstrategy
dynamically export jvmci without using a provider of an exported jvmci service
jsonwriter minimal output type has to surround strings starting with numbers to differentiate between actual numbers
substrateconstantreflectionprovider readfieldvalue must not propagate illegalargumentexception
improved channelfuture documentation
fix perfect reflection on collision
no logical change add missing copyright notice
fix compile error
fix test assertions for fast cloudbees
refactor rename avoids hiding a field
null check on share
add czech default stopwords into named stopwords map
regression in interpretercallbenchmark directcall
used subscribeon instead of observeon
renamed zoomcenter to zoomtocenter
some javadocs ci skip
add println when all the tests pass
hotfix coordination log if taskexecutor establishresourcemanagerconnection fails
error stream isn t always available
merge branch master of https github com petehayes hudson into petehayes master
fix a couple typos in sl javadoc
hotfix minor cleanups in jobleaderservice
revert change to placeholders in config file initializer
create the output file if it doesn t exist
set the thread pool size only when the configuration is changed
modify the seek
fix unresolvable symbol in javadoc
stupid gwt inherits
config cleanup work
adapt to renaming of resttestcandidate
fix crash for dynamic permissions background directory load
add new constructor and setter for vector3
add a test
remove unused imports
rsv status label redesign
test subclass elasticsearchtestcase in loggingconfigurationtests
add the message recevier log
modify the manager
support datum is null case
add fixme comment to assetmanager remove need to think about that
sulong assert that traceloaderstream is properly initialized
round up managed malloc sizes rather than calling it an error if they aren t round
fix coverity bugs
make field comments javadoc in optimizedosrloopnode
remove null check for proxycontext getruleregistry
test use the correct translog setting
add httpmessage setheader string string
when the key of map set is a custom object you must override hashcode and equals
code and tests for assertj
oops seems like javac chokes on the r while eclipse jdt has no problems with it
filters custom value fix
uncomment expected exception
test now that we check early if we are corrupted this can be run with many nodes
fix looping in animationfragment
sql completion check
show enabled property on pks
update levelordertraversal java
including hibernatesessionflushingtests in the suite
fix streamingdrmsessionmanager use locally bound variable instead of class variable
fix checkstyle error
avoid class cast exception
mute failing test
remove superfluous null check
idealgraphvisualizer add textual representation and coloring scheme for the graal compiler and remove maxine specific code
better tostring for llvmtruffleobject
fix javadoc comments to match behaviour
use object readresolve in api token properties
relax wrong assertion
add missing copyrights
test wait for green until testing version force
redissoncountdownlatch race condition fix
modify the service
remove an unused import
fix language classes need to be loaded eagerly for aot
document new buffer unsignedxxx methods signed off by thomas segismont tsegismont gmail com
minor fixes for javadoc warnings
remove commented out methods
add warning message when using the onredeploy option signed off by clement escoffier clement escoffier gmail com
fix invalid whitespace
sulong fix npe in vectortype exception message
merge branch master of https gitee com linlinjava litemall
documenting why this can be null
deleting commented code
add spring data redis serializer support
deflaked test of semaphore concurrency
rsv status line color fix
workaround for impossible cast warning
add assertion data is stored as string signed off by paulo lopes paulo mlopes net
bug fix for encodeutf8
test use more verbose assertion in indicesrequesttests
fixes failure to transmit interesting unicode characters by specifying http utf_8 in urlencodedformentity
default names converter fix
clarify where datastream print and printtoerr print to
made classes static fields final
remove unnecessary logging
add buttongroup getbuttons
make sure the pubsub listener is subscribed when a connection is reused the listener should still be subscribed
hotfix fix checkstyle violations in zookeeperutils
hotfix core fix javadocs for temporaryclassloadercontext
data transfer metadata create
add javadocs to describe serializedsubject class also correct error in javadoc comments for operatordoonsubscribe for what it s worth
disable view prefetching for now
update frame verifier to support new scheme for byte arrays
merge remote tracking branch origin master
hotfix remove unused total_flink_memory_mb field
pg temp table ddl
clean up string format it works like printf why not use it like printf
fix potential npe use discardreadbytes instead of clear
use custom thread class to set priority properly
allow codegen on embedded vertx factory methods so vert x can be run embedded too from other languages
add new intrinsics
performance issue when handling strings improve performance by using stringbuilder instead of stringbuffer
s xxx todo
update maths armstrong java
add a log field
use fstconfiguration deriveconfiguration to preserve conftype signed off by chris eager christian turo com
fix array type in ifnode constantvalues
don t insert character zero if a font actually has character zero textfield would go nuts
disabled printing of inlining decisions
graalcompilertest throws an error if code installation fails
fix cachinglist clear
make view better
new comparable value
add restrict all dir to native image agent to read all configuration files in a directory
fix memorytablemeta processtableelement bug
asserts to check overriding of options
jdbc ds reconnect fix dup contexts addition
minor fix for displayed estimates
add ability to create materialized frame to truffle runtime
fix annotation errors
hotfix tests fix topic name in flinkkafkaproduceritcase testscaleupafterscalingdown
mongo functions system collections
merge branch master of code dianpingoa com arch cat
remove call to rxjavaplugins errorhander in mergedelayerror operator compositeexception is already visible
prevent oops when whitelisted commands input is empty
re introduce and deprecate bdpd initdefaults element for compatibility with spring integration s standalone usage of beandefinitionparserdelegate
connection dialog fix general page npe
delegate componentscanbdp environment to scanner enables profile filtering within componentscanbeandefinitionparser
update stringtobytearrayunittest java
delete import org apache shardingsphere shardingjdbc orchestration user yamlusertest
grid remove unused imports skip ci
remove unnecessary warning logs related with clientbootstrap
only nested libraries contained in the lib directory in the pact jar are add
test bw comp testmultiget should wait for yellow not for green
rsv presentation activation fix flickering
fix missing truffle boundaries in tests
use empty string for absense close reason
remove unused method hystrixcollapserproperties getproperty string hystrixcollapserkey string string string
use type size instead of constant
make methods accessible
use tools instead of instruments in help output consistently
streaming license add
java deleting unused local variable
revert jooq application java
consider groups unknown if they have no title avatar or members
revise springnamespaceexample for xa
dialog fix another stupid npe
mark messy rest test as messy and add to messy tests lists
grid tests throws is not needed skip ci
fix copyright header
url preview editor fixes
enforce method flags during interface method resolution
remove loads of warnings
register the new settings
folder switch fix
renaming to hostaccesspolicy also in graal compiler suite
fix animationtest gwt compatibility
fix overly restrictive condition to identify disjoint plans
mute partitionedroutingit testshrinking on windows
add methods to get file names and dependencies an asset might have
fix truffleprofilertest test hook is not initialized eagerly
comment regarding synchronization in cache computeifabsent
fix missing import
java unignoring back a test that works well despite the fact that the target issue is not closed
tests replace constructor in proxy test with static before all
modify to use startindexofappendedparameters
fix stub image using
restored preference for covariant return type if applicable
add public methods
add assertion to contexts be contained in the contexts set for preinitialized contexts
externalcompilationresult gate fix
delete invalid functions
alignment bug fix contributed by mick jordan
add tostring to nodestack
updating javadoc of plugin labelatomproperty globalconfiguration with my observations from working on my plugin
hardcoding sl mimetype for now should have an api so language implementations can declare their own supported mime types
relax assertion in emitnullcheck to kind object to allow for illegalstamp
hotfix test do not hide original exception in the sourcestreamtasktest
db instance model
add javadoc to util java
more string trap
make sure splitifatphi is not appliest before fsa
remove null check and unused imports reformat comment a bit
hotfix tests fix manual tests to properly start testing cluster
fix missing import
fix typo in test name
sulong make syscall translation table compilationfinal
remove unused code in currentjavathreadnode
sparc control flow now for at bt ae be
always configure npn without this we get a hard ssl crash when a server returns npn information that was unsolicited
add renew function
adds debugging information for invalid repository data x content
polish javadoc for abstractbeandefinition
more robust unit test
javadoc for typeadapterfactory in jsonadapter
rsv scroll fix keep selection
ci bump copyright dates
oops there are two checks in this test only one was fix in the previous commit
make sure we set a classpath resolver for a verticle deployed from classpath
fk edit page fix update buttons state
fix zero or negative time interval not supported
changed type to name to avoid confusion
cronschedule moved to org redisson api package
replace getdefaultdriverclass with getdefaultdriver format code
actually close websocket
remove unused import from o e e environment
update gridfstemplateautoconfigurationtests java improved documentation
fixes per content review
vminspection isenabled add
set sulong language id to llvm
modify the code
fix existence query bug again
check package name for extendedsocketoptions
error text fix
add a getheader method with a typedkey in restrequest
more javadoc fixes
merge branch master of https github com libgdx libgdx
fix after review no need to ask for service_name twice
rs filtering fix keep original rs metadata after add filter
bug format type confused
remove some no op calls to deprecated functions
hotfix let streamgraphgeneratortest extend testlogger
fix checkstyle nits
fix string index bug
new test case for takewhile that currently fails
trim whitespace before trying to parse
truffle improved astsize output for opt done events
allow null from the name
explicitly do not use jna
also round background drawable if colordrawable or bitmapdrawable
do some little cleanup
mysqlvalidconnectionchecker use pinginternal replace ping
better diagnosis when resolvepluginfile would be about to return null
fix typo in loggingapplicationcontextinitalizer initializelogleve initializeloglevel
eliminate bunch of exoplayer warnings
add problem source url for mischeviouschildren
fix visibily of immediateexecutor and add private constructor
rotate the cds feature flag
add firefoxdriver constructors that accept legacy xpidriverservice
add symbol for aot
new shardingproperties new properties
fix style problems
updated log statements to be localizable
remove unused imports
null is a valid input to a guardphi
fix initial countdownlatch value in test
add dcl statement judgement
chain the root cause this allows hudsonauthenticationentrypoint to access which permission was needed
hotfix network fix codestyle issues in nettyshufflemaster
polishing formatting of dockerapi
modify java doc
recent lwjgl versions allow glbufferdata calls with null data
fix npe on latch in test
improved javadoc for management attachment
focus on rsv when sql editor tab switched
add a warning for merge paths
add corrent tostring and json output for data sinks
remove unnecessary call to seekto
remove unused import signed off by stefan marr git stefan marr de
documented the implementation of jsr ret verification
add javadoc to asyncmap about null keys and values signed off by thomas segismont tsegismont gmail com
make atomicboolean final
remove magic number in macro assembler
add httptunneladdress tostring
add nonstatic observable finally0
simplify ingest useragent construction
use correct workspace the one use for the build to get settings xml
fix stream decoding
modify the gzip
fix hibernate cache shouldn t be cleared on destroy
add the request id to the timeout message
fix android q storage
update encoder java
fvh should use the parsed query and not a possible rewritten query
fix comment about when this is set
attempt to fix windows ci
truffle dsl fix a bug in elementutils gettype
fix stackoverflow error on internalnumericmetricaggregation
fix add support for rgb888 to texturepacker
java fix heuristics used to detect test execution root directory
make code less
truffle dsl fix crash for nodes without rewrites but with caches
omit useless argument
quiet means quiet else the real exception can be eaten some streams fail in more spectacular ways when closed
renaming to imagegeneratorthreadmarker and add a warning into its javadoc
possibly better compatibility though does not handle someone actually overriding parse
disabling an unstable test in chrome
reset attachment transfer state if auto download requirements are not met
rsv confirmation fix ui thread
merge branch dev metadata of https github com apache incubator dubbo into dev metadata
connection test fix features info
change routeunits to final
fix for a ripple effect on a button
move package declaration
allow calls to jacocoinit from clinit in a class declaring an option
more fix of broken tests
fix npe in fts when snippet is null
fix missed test class
seek duration should not be rounded up when opening the settings even if inexact seek is disabled
add some extra ouya mappings
make language agnostic inlining work when inlining is disabled
add defense to selectnewpathforshard
work on scaling and blocking thread pool
add annotation override to com iluwatar reactor app appclient tcploggingclient run method
fix bug in vbo va test seems like binding a vertexarray with an empty direct bytebuffer is a no go
take care of missing for file path
java disabling a couple of tests in chrome that look like regression bugs
mute mlconfigindexmappingsfullclusterrestartit testmlconfigindexmappingsaftermigration signed off by mark vieira portugee gmail com
fix copyright text in wasmloopnode
clarify variable name in indexingmemorycontroller availableshards
remove hardcoded index when parsing playresx and playresy
make com oracle svm driver nativeimage accessible
add usage to javadoc for breakpointnode
remove dead code
ruby flavor fix logic issues in p escape parsing
merge remote tracking branch origin master
use correct test address in httptest testworkerserver so it works with native test profile
fix style warning
don t drop metadata annotation in newinstancestub
make barchart more extendable
add get method for hintmanagerholder
add factory method for null pointer llvmtruffleobjects
remove unnecessary declaration of testexecutionlisteners
tolerate non lowercase macrooptionkind names for now remove once all native image properties files use lowercase
methodhandleplugin fix calling native methods
fix javadoc error
to fix a compilation issue
rename to eliminatednodenames
add comment to clarifiy why metadata fields can be set all the time to indexrequest in pipelineexecutionservice
make guardingpinode canonicalize to a deopt when the guard is guaranteed to fail
some more testing for access checks implemented reflective interface method calls
language info instance needs to be initialized to be a valid initialization of the rootnode
java making testgeneratekeypresseventevenwhenelementpreventsdefault less strict to possible mouse movements
modify the code
make a cpu intensive test pass on a slow machine
remove useless code
disallow punctuation in icon initials
add deprecated mark
resolved a merge failure
remove unused import
add framemap getter to lirgenerator add compilemethod overload with input graph parameter
updated test names to reflect what it is testing
oracle view existence check fix
do not leave interactivebreak in committed tests
pluginmanager rename xpack plugin to x pack
enable trace logging for randomexceptioncircuitbreakertests testbreakerwithrandomexceptions
tabbed folders fix
data time value manager fix edit enablement
add context to file upload handlers signed off by oscar albrecht oscarfh gmail com
remove unused inject methods
update rule javadoc
fix session issue
node cost util rename function
fix missing double dash in warning message
fix compilation problem since we dropped registertypehierarchyadapter method
remove needless settings from transport client
more efficient to contact crumbissuer just once and get both name and value
make the cds flag remote capable
add a system gc just to be safe
java remove misleading serializable interface from capabilities because they are not serializable
test fix explain api test to allow for either awaiting info state or no valid shard copy
hotfix minor javadoc correction in statetable
an experimental workaround for infinite blocking sslengine
remove leftover debug code
remove superfluous repetition freebie
invalidate on out of bounds memory accesses
test plugin manager undo removal of change
fix start submitdebuglog from registration and passphrase prompt
add mouse scroll to scrollpane
remove accidentially commited code
fix rocketmq ack npe
make fillerobject immutable to avoid monitor field
fixes npe in requestfromchild method
compute the isprofilingenabled information everytime to accomodate possible change in the truffle runtime
merge remote tracking branch origin devel into devel
java copying static resources should create target directory
show full stack trace
test fix assert
fix typo in class name
fix basedrawable setminsize incorrectly using minwidth for the minheight
for java doc
rename to yamlauthenticationconfiguration
add assert abstractstandardrequestupgradestrategy
allow to disable traceclassinitialization
read only columns
some javadoc for getxstream
test compile fix
introduced a small hook point for subclasses
om add default implementation of new methods for backwards compatibility
tinker tidy up demo codes
reference hotspot implementation
modify the config manager
fix comment formatting in evilloggertests the comments here were formatted inconsistently so this commit fixes them
support type level responebody and restcontroller
hotfix table common enable hashcode equals for materialized raw value data
remove unnecessary javadoc
missing one xaxis entry
add error detail info
better tostring on transport future
fix wrong angle direction
filter out dead nodes when filling a nodeworklist
add broadcasttableruleconfigs to shardingruleconfiguration
node costs fix formatting issues
update avroserealizerdeserealizertest java
updated gwt backend emulation with joint user data capability
hotfix typo fix typo in containeredtaskmanagerparameters
modify statement checking order in sqlparserfactory
cdp throw an exception if unable to process a message
clarify that exits happen as soon as we know control flow exits the loop and link to example
fix invalid reference in javadoc
fb annotations for trigger class
merge branch master into annotation newoptimizer
temporary fix to transfer end of queues to hard disk
changed executor service in jobmanager to support larger clusters
reenable usage dropped to zero listener
fix base address in data section
set default reason
remove scanner close at multipleofseventeen
fix test to expect assertionerror as well
hotfix set initial bestcandidatescore in locationpreferenceslotselectionstrategy to double negative_infinity
add initialization on demand holder idiom
fix optimization to not do sorting when searching with score desc sorting
make the graph object always available in context helps debugging problems during lirgen
fix to condition implies
let stampfactory ortypes use riresolvedtype leastcommonancestor
re enable redeployer
parser make magic version constants uppercase
disabling flow sensitive reductions until their issues are addressed
add comment explaining insertion of nop during commoning of constant loads
dbe configurator refactoring
make all duration numbers less
add licence info
java simplifying previously merged pull request
fix package name
fix compatible for old version java serialization
fix documentation error in client server bootstrap
add scalar as keyword
mentioned parameters compiler flag in javadoc
call ensuregreen to get more meaningful debug output on failure
bug fix for pgvalidconnectionchecker
java fix edgehtml driver initialization in tests
tracera fix format
move padding out of the for loop to avoid unnecessary calculation
remove double in the url though seems like couchdb does not mind
make sure previous counter is never negative
also support camel case notation for minimal norwegian
fix small bug in encodedimage clone
add a log for reloading jenkins
remove trace logging from coordinatortests
rs cells copy fix
replaced broken lt with lt and same for gt
actually count number of bytes sent in protocol handshake
increased time wasted
cancel okhttp3 requests
modify shardingonlywithdatabasesandtables java
fix probabilities in expandlogicphase
rename fromextractresult to fromclauseextractresult
remove debug println
hotfix runtime minor improvement to logging in checkpointcoordinator
save parameters types
add missing transfertointerpreter
hotfix remove duplicate lines
fix invalid error message assert
truffle compiler put arguments in message of npcnode
get rid of superfluous queue task executable
minor fix to improve robustness of byte buffered channel manager
fix javadoc again
regex escape resource paths from the agent
remove trailing whitespace from liteblockingwaitstrategy this was causing checkstyle to complain
merge branch master of github com mrniko redisson
currently the system variable to chose which type of zlib decoder is being negated
register classes exposed via getpublicclasses and getdeclaredclasses calls in reflectionregistryadapter
hotfix network checkstyle
navigate to parent uploader channel from the uploader section on long click
switch example package to sample
add exceptionhandler hashcode to remove eclipse warning
small fix to postordernodeiterator
check nojdkzlibencoder before comparing other parameters for consistency
remove some debug code in hotspottruffleruntime
fix a code style issue
fixes issue for locating newly installed chrome versions signed off by luke inman semerau luke semerau gmail com
checkfornull as suggested by stephenc
return empty scopes even if env is null
fix potential invalid context crash
fix compiler error for code merge from android version
don t use deferred initialization barriers for native image
polish javadoc in exceptionhandler
fix a buffer rewrite trap which occurs when directchannelbufferfactory is used
prevent a crash when there is a shape with no points
make the llvmreadvisitor final
use correct color value in imagedecodeoptions builder
mediatype parsemediatypes now accepts null and
add neverinlinemacro for optimizedcalltarget call
tracera fix fixedinterval tostring for endmarker
fix npe when importing markdown files
ensure clock adjustments does not stop remote config refresh
expose the remote cause details
use linkplain for text links
enhanced assertion message
remove obsolete test
modify the handler
fix invalid visiblity for inherited object
call onnewengine before the engine is actually published
fix nan comparison error
fix closed connections handling
put loopunswitchtest in the right package
assertion in node
obtain table name from postgre rs metadata
hotfix mesos fix the access modifiers in launchablemesosworker
add implicit string type for typed string values w o a specified type
correct error in setting cu index for deopt targets
fix rogue license header
fix server error path reference in error controller
missed my period
redissonmap clear optimization
hotfix runtime fix mistake in restfulgateway and jobmastergateway javadoc method requestoperatorbackpressurestats never returns a future that completes with null
ignore non grouped tribe settings
fix getconnections bugs for proxy
profiler javadoc fix
grid add missing event bus role to message bus command
only allocate storage for an intern global variables once not every time there is a reference to the variable
tests did not match naming pattern so were never being executed
actually updating capabilities with fix noproxy
extract policy as a field to enable testing different policies
om dsl annotations can have source retention
truffle remove unused code
improved error diagnostics
skip invokes with a placeholder framestate in methodcalltargetnode simplify
sparc add hint for move result
do not assert for mergenode in usetrappingnullchecksphase
change prefer direct for allocator to true
editors pref page fix
erd refresh fix
fix effectively set blended tiles
add installstate override
send returns channelfuture
erd load fix npe
fix typo in abstractfieldmapper similariry similarity
move circuitbreakerstatement java
suppressing some known good instances of an idiom otherwise warned about by netbeans plugin for jenkins
config the transaction analyzer
schemastatvisitor improve support mysql
no need to pre initialize the debugger with debugger find enough to call executionevent getdebugger
split fluxprocessor factories into individual processor remove confusing factories
documentation on writecomplete event type
invoke uncheckedstamp return type of invoke is already available on methodcalltargetnode
changed configuration of checkpoints
hotfix expose abstractfscheckpointstorage resolvecheckpointpointer as public
implemented daniel beck s suggestion for defense against build number duplication
fix source findlanguage was failing if engine was not yet initialized
add location canset value
fix modsuffix ternary expression so it appends mod when true signed off by david dossot david dossot net
remove test for current localisation as it was failing at prod due to different systems
use variables if they exist
renamed conversiontest to canonicalizedconversiontest and improved it
update since tag for streamutils
tachyon minor fix in codestyle and warnings
regexresult no longer contains a field called groupcount
legacy polyglotengine uses should default to allow native access
fix body destroyfixture puts fixture back in pool as well
remove orche tests
fix the re apply of rounding
remove unused code
add since tag
remove logging statement when no shards are active
core fix duplicated imports
fix eclipseformat issues
append global stub address when disassembling a call to a global stub
rename field to be consistent with name of accessor method
reenabled forced software rendering because it makes some animations janky
add new tostring
editors auto save redesign
hotfix cli update the help message of the generic cli
hotfix zookeeper log changes to schedulingstate
sql editor log filter fix
fix a typo in an error message should exists should exist
if not set the context path with the webserverinitializedevent then real is context path
save password workaround
stackslotallocation improve javadoc
solve compile error
add jsonp test
few comments add
commented flyweight example
fix checkstyle violation
enabled execute evaluated methods in typed node
simplify an if else
fix bufferutils size never decreasing
add some type to transactionoperationtype
idataset interface cleanup
hotfix make slotmanagerimpl findmatchingrequest and findmatchingslot private
add blank line
small optimization to counting search shards
return lastinsertid for dml
fix typo in comment
add reserved word tests for doc too
remove on liner method used in only one place
remove unused field
fix broken test in annotationutilstests
sulong resolverenamedsymbols fail if symbol not found
grouping check for subselects support
test case created for hextoascii
delete unused import
fix typo in truffleprocessbuilder constructor
merge remote tracking branch origin master
minor fix in large valueformatter
be defensive detecting when mono not running
hotfix fix directory creation in typeserializerupgradetestbase
add todo note for result validation
add missing import gdxruntimeexception
swt bug fix
fix after review
rename result to extractresult
remove beginnode from abstractblock interface
fix imports for javadoc
fix possible nullpointerexception in hotspotmethodresolvedimpl tojava
linearscan remove superfluous islogenabled
align foreign call descriptors
regression close player in onplaybackshutdown
add a convenience method
used correct shipping strategy for reading data in partitioner
modify the record
checkstyle errors fix
unbreak group insertion for new installs
add basic banner to cli shell
modify the logview uploader
print statement only if debug is activated
remove log for debug
mute failing getresulttests and documentfieldtests
remove unnecessary try catch block in abstractunsafe flush
suppress warnings gradle build
internal remove unused constructor and adjust methods visibility in delegatingactionlistener
plan viewer render fix
crap merge error
add javadoc for response body
qm listener fix sessions reopen
fix datasourceinputstream read for eof cases
add exception to mutable object iterator interface
fix some more tests
clarify plugin documentation for requiresunpack
remember me option doesn t always show
fix chaining in overloaded methods
we must still initialize and clean up the legacy trigger timer
relaxing the rule of always having some local scope
webdriven moving a test aux class to a more appropriate package
fix implicit int to long cast issue
deleting unused imports
cache method pointer stamps
add getters for various jni data structures
lirgenerator remove setcurrentblock
fix no idea really
fix a problem where the local transport does not fire a writecomplete event
delete import java sql connection
reinstate required parameter
merge branch develop of github com alibaba nacos into develop
make transportrankevalaction members final
test thrown error when there are no rest tests to run
remove debugging code and redundant gating sequence
fix a potential npe
add a fixme for later releases
fix import issue
better docs for newsessionpayload stream
update media timestamp regex to allow negative values
modify the state analyzer bug
fix spawn node to insert call node in execute method update test assertion accordingly
update name used to monitor the number of fds in hardwareconfigstate
update scrollpane java
browser view layout
fix bug in datasourcebuilder
http compression level signed off by stefano gargiulo info creativeprogramming it
sllanguage visualizer is initialized statically remove lazy check
java add double to return types list
the datalayout for llvmcompareexchangenode should be given by the nodefactory
merge branch master into code style
turn checkstyle off for the original copyright notice
command execution attempt logging add
fix reachability check to allow fix in serializearray specialization
bugfix stamp of fixedvalueanchornode cannot be nodeintrinsic stamp because it is instantiated via constructor too
add props to pass parameter
don t cache bytesref in threadlocal
fix the failing grid tests
blockingobservable tofuture cancel races with onsubscribe
fix nullness for not equals in conditionaleliminationphase bug reported by miguel garcia miguelalfredo garcia epfl ch
speed up execution by having an early heartbeat
hotfix tests fix the missing class reference in localinputchanneltest
register rax as a temporary for frem and drem stubs
lsra fix getmaterializedvalue respect musthaveregister priorities
fix expcetion msg comparison
unwrap unsupportedspecializationexception from an interopexception in symbolinvokerimpl and re throw it some unittests e g in sl expect unsupportedspecializationexceptions
fix issue in abstractmessagingtemplate
fix typo in listener spelling
fix annotation javadoc
correct the stamp on the pinode from a range check
tinker avoid odex checking on meizu devices since it would not generate odex when load dex first time
allow null for profileavatar in retrieveprofileavatarjob
delete some properties
fix a bug where head response is not read completely if it s transfer encoding is chunked
trying to fix a build
update alltests java
change default namespace name to application
delete set function
fix some warnings
don t save local data using diskcachestrategy all
graphdecoder add debug information to invoke assertion
replace all occurences in the class file
tinker fix build failure
delete obsolete interface
properly report failures to start the safaridriver hopefully this will eliminate the cryptic executor has not been started yet errors
review fix changed the order of arguments
rsv gradient colors for whole line fix color for out of range values fix
fix scrollpane padding
fix security crash during directory refresh
fix npe in oioworker write when the socket is closed already
remove stdout call
improve instancehealth add update logic
fix hostlanguage should allow accesses from any thread
remove unnecessary string valueof call in stringencoder
fix style check failure
clean up warnings in spring core
move knowledge of json parsing into jsoninput
change codahale to dropwizard in javadoc
modify decrementkeygenerator java
ignore test for correct browser and add reason
clean up more todo comments
needs to be visible from the subtype
content editor panel menu fix
ignore a test case with timeout it sometimes fails on travis recently
reset the master ping counter when we get a proper response
hotfix set rescalinghandlers timeout to rpcutils inf_timeout
throw instantiationerror when trying to instantiate an interface or abstract class
also add jenkins
fix visit loaded source sections
hotfix coordination remove unused code
update proxy password
truffle simpler more consistent tostring for directcallnode subclasses
fix wrong statement in indexbufferobjectsubdata
fix date value
modify userrepositoryimpl java
lookup tag based on index in enum value array
adjusted regression test duration
hotfix test fix streamconfig propagation to streamtask in streamtaskmailboxtestharnessbuilder
fix coercion spelling mistakes in javadoc
we don t need to check this
fix bulletobject needs to include stdio h
add ignore flag
connection ini error handle fix
small fix in graphorder assertschedulablegraph
use only recognized flags for init modifiers validation
rename to isowncompleteconfigurations
findbugs handle non realistic case when null staplerrequest gets passed into defaultcrumbissuer and maven step instatinations
fix unhandled_prompt_behaviour capability to work in legacy browsers
fix possible context crash
improve simpleassemblertest junit assumption
comments for state example
should throw sqlexception on failing getconnection
put synchronization behind truffleboundary
fix fix code style
close stream on exception
remove unused import
prevent recursive inlining in llvmsourcedecoratortype
fix buffer leak in abstractmemoryhttpdata
use the right class loader when initializing sulong
add in doc the supported pem formats
modify the nullpoint exception
add leaf snowflake key generator and related functions
bugfix can not navigate to tatget if there is no any interceptors
deleting unused imports
databoundconstructor makes no sense on an abstract class
add an explanatory comment about the timeout
java changing simple assert to a wait
fix the interceptor class of jdbc plugin is incorrect
internal call context handletermslookup rather than context indexqueryparserservice handletermslookup in termsquerybuilder toquery
truffle turn milliseconds in nanoseconds for the time threshold after options have been considered
remove matchers wildcard
yuck hack around snappy printing to system err the failure to load its library if its not found
fix a bug where allowing jsonobject innermap to be put non string key
fix potential conditional npe
merge branch scanner of https github com jkreindl sulong into scanner
update javadoc for testcontextbootstrapper webappconfiguration is no longer meta annotated with bootstrapwith
fix filelinereadwritetest unit test
weaken the assertion in source equals for now
verify number of arguments to foreigncallnode matches signature
don t wrap in broadcast shard excpetion twice
stringbuilder fix broken append
put ptx logging behind the g log option
hotfix checkstyle fixes in memorymanager
remove unused code
add promised comment to testcanresetunreasonablesettings it got lost in a broken git song and dance
comments on performance increase
ui utils fix use default shell display if no workbench windows is active
tinker enhancement use bufferedinputstream when reading dexes from old apk
test reduce size in bytesizevaluetests testequality to avoid double rounding issues
save after calling setsecurityrealm or setauthorizationstrategy
don t walk out of loop when estimating unswitch benefits
update copyright year for changed files
fix one more timezone related test failure
update core src main java hudson extensionlist java co authored by oleg nenashev o v nenashev gmail com
fix no such alert error mapping
stop proximity sensor on pause
update binarysearch java fix
buildstep prebuild needs to be called for post builders as well
check if sherlockfragmentactivity is used
do not fail hudsontestcase test when testenvironment dispose fails jenkinsrule does the same for years
set query cache to null
oracle structures support
fix memory leak
hotfix fix continuousprocessingtimetrigger
truffle share frame descriptor among cloned call targets
fix compiler warnings
canonicalizer trycanonicalize return early if the custom canonicalizer has simplified the node away
fix an issue when iterating over node positions
mysql tools fix specify cl password only if it present in connection
make argument ordering consistent across web server initialized events
improved name of instrumentation graphs
default log level if it s missing
fix bug of diskcachestrategy resource this bug will cache source data at diskcachestrategy resource
deprecating jenkins model configuration
lirkind verifymovekinds relax stack kind comparison
fix test failure message
unignoring a test fix in firefox nightly
disable test when running with compile immediately
null parameter checks in bulkprocessor builder builder
fix a issue for isdone methond is not work
add oracle rownum test cases
remove linkage to slow path monitor stubs which were only used by xir snippets
cruddocumentationit fix documentation references
only tear down isolate if isolate compilation is enabled
change monitor rules to original
clarify naming for clearkey drm support
add problem source url for basicallyspeaking
more node type fixes
log blob cache and server storage directories at startup
add missing copyright header on cursorexception
missing version tag from single concatmaps
merge branch master of https github com didichuxing doraemonkit
merge branch master of github com alibaba arthas
revert action state fix
compacting switch block
revise java doc
rsv gradient colors fix no coloring outside the range
modify the server config
document that child and children imply compilationfinal signed off by stefan marr git stefan marr de
fix for the method handle plugin
merge branch master of github com eclipse vert x
remove generic warning
this shardingproperties shardingproperties
add a link to channelconfig in channel
continue remove gson from the json package
use scheduled delay instead of manual one
modify incrementkeygenerator java
rename torture test to stress test
exclude from animal sniffer
change defaults in stress test
fix race in deployment test and remove the deployment from the deployment map when the deployment leave the undeployed state
call super finalize per sonar when overriding finalize ensure a call to super finalize at the end of the method
remove installmethod from rimethod
actually export the firefox builders correctly
fix configtimeregionfactoryholder reset
enabled checkpoints by default
add log info
fix intermittent infinite loop in abstractoiobytechannel doread oiobytestreamchannel doreadbytes did not expand the capacity of the inbound buffer properly
use appendrest for removetoken
lookup threadmxbean lazily
remote duplidate space
fix compilation in eclipse eclipse needs a bit of extra special help with type parameters in transportreplicationactiontests now
test fix possible npe in clientyamltestexecutioncontext
fix post alert text command seralization
for an unknown reason a different exception is raised in nioworker read
for code format
this test turns to an infinite loop here ubuntu
optimize messagelist add messagelist int int
gave utility classes private constructors
sulong put truffle boundary for blacklisted svm method
commented factory method example
performance fix to avoid reflection based initialization of null key surrogate
rename insertionsort java to sorts insertionsort java
core show warning when failed to detect out node in non trivial switch
made barrieredaccess final
hotfix fix javadoc typo in localfilesystem java
oops forgot to rename a usage
merge remote tracking branch origin master
simplify astutils getpropertyaccessorstotry
clusterconfig check in loadbucketvalues add
fix for creating non push groups
add problem description for factorialfrequencies
made edit profile save button move with the keyboard
tinker bugfix npe when isprotectedapp option is enabled
checkchannel provides a better error message in case you are not using remoting mode
avoid silent overflow when growing jni handles storage
fix proxy api tests that were expecting an assertion error
handle null exception messages
fix glyphlayout using lineheight rather than down down is negative hence the subtraction
fix recovery debug logging param mismatch
remove redundant field initialization
add missing license header
fix some clone methods
fix redissonconcurrentmaptest remove two unnecessary lines that s causing the problem
order switch style
work around weird gb layout limitations that was freaking annoying freebie
remove unused import
truffle source fix bug in linelocation comparison
correct misleading warning about firefoxdriver constructors
add unit test
dynamicnewarraynode add exclude for jacoco
fix a test bug
remove uncompleted feature
rephrased the comments on compression handlers
deprecating computer clionline functionality was moved to onlinenodecommand class already
only try to convert to string if canconvert returns true
update ivorytower java fix the issue with class creation
temporarily disable failing guarantees
fix typos in javadoc
message properties is a known one
change order of properties in endpoint mbean objectnames to make display in jconsole nicer
perform deoptimize to guard conversion for fast partial evaluation
add call timeout option to okcurl
add glyph tostring
fix small error in distance normalization in test
add empty constructor in order to work with json
fix test bug lastfollowtime is null if there are no follower indices
use a uridatasource for dash manifests this allows loading a manifest from a file uri
add theme reloading api
small optimization for global facet with facet filter by remove the double check on the facet filter
increased the timeout value for commands to accommodate travis slowness
fix the build
fire change event in setitems
fix sparc unused annotations
remove unused imports in o e c a s shardstateactiontests
merge remote tracking branch origin devel into devel
java deleting an unused private field
reference aes spec to show block size is constant
fix reset native image home did not properly reset the nativeimagecache
add vector3 epsilon testing
test additional logging
fix bug in defaultdeliverystrategy signed off by thomas segismont tsegismont gmail com
changed control scheme
suppress exception in finalizecontext
color functions which clashes with keywords as keywords
update multipartfileuploadclient java
name logged fields in eventlogger
execution context refactoring
tighten up scope of suppressed warnings
add blank line
fix problem with missing thread interruption in class replaythread
move encryptcolumnmetadata to encrypt module
netjavaimpl fix previous commit
add missing copyright headers to java source files
cull frame states before lowering
fix button ripples in videodetailfragment
change virtualstackslot tostring
fix returnlength assertion
tweak javadocs in sprinapplicationbuilder
rename to assertgetdisabledslavedatasourcenameswithdisableddatasources
add more wait for green status on aggs tests
set http on nodeinfo instances if it is passed in the constructor
make imageclassloader robust against invalid jar files on the imagecp
setbackground null has caused layout to npe
fix missing boundary in dispatch libraries
private properties keygeneratorproperties new properties
fix equals method
fix missing urivariables
only prompt for importing system sms db when we re the sms app freebie
fix invalid import
fix wildcard imports silly intellij config is being overriden at the moment
hotfix tests remove legany codes from checkpointbarrieralignertestbase
fix doc link
fix import ordering
fix failing stylecheck
remove line break
fix query limit use unlimit instead
gp model fix
remove an unused method in readbufferpool
add logging on which state file failed to load
attempt to fix failing log tests
add javadoc for programinformation
instrumentationtestlanguage make a bit more friendly for tool development use by giving a name that is easier to type
suppress deprecation warning in statusresultmatchers
expose jenkins in multi cast dns
don t call deserializers with null
change log level message
rename to assertgeneratekeybeyondmaxsequencepermillisecond
put stack allocation behind transfertointerpreter
fix double calculation of xaxis spacing
fix user option value
documenting the behaviour
fix typo freebie
fix compile failed
fix build failure
fix stamp in unsafe load lowering
simplify new internalerror initcause e to new internalerror e
active schema value fix
fix spelling mistakes
ignore when flush can t be done
try check again
move insertcolumnstoken to pojo
fix broken build
fix compilation after incorrect merge
spillmovefactory add createstackmove
use constants when caching empty arrays reduce cache memory consumption by using a single constant for empty arrays
remove redundant commit opentranslog already commits in that case
fix lose alert bug
don t throw index missing when updating number of replicas on a closed index
remove duplicated registration of warmers as custom metadata type already done in a static block within indexmetadata
java fix coercer ability to access private constructors
accept all pointers in toi64
fix localnode dumping in cfgprinter
add pattern case_insensitive for mysql
merge branch master of github com mrniko redisson
small javadoc fixes
unsafecast node is nonnull if it casts a non null object
remove superfluous method
document the need for suppressing jarfile caching
fix typo in javadoc
hotfix javadoc rocksfullsnapshotstrategy remark description correctly
fix wat2wasm system property in wasmbinarytools
support other stamps in constant forprimitive
better naming convention
fix for detached illegalstateexception
lirtest remove simplifiable from floatinglirtestnode
if canon allow same direction being two loop ends to the same loop with the same phi value
move second removevalueproxyphase
add apache license header
profiler fix bug in timedcounternode use of frame slots
add thread safety comment about ensurecloned
add framework for two tests in scalar model
fix compiler warnings
small interface fixes to suitesprovider and monitoridnode
fix incorrect comment
modify the phoenixagent
hotfix metrics make jmxmeter class static
java deprecating okhttpclient prior to deletion
flow sensitive formatting
fix package declaration
unwrap invocation exception on reset
format the mess codes
truffle add getqueuedcalltargets to graaltruffleruntime
fix issue where sms groups showing up as unnamed group
switching to beta at daniel beck s suggestion
fix issues after merging in master
remove circuitbreakerconnection java
getorchestrationfacade getlistenermanager initlisteners
java deleting redundant fixture initialization
hotfix tests fix that invokableblockingininvoke can finish from spurious wakeups
tweaking the wording a bit
for java doc
if the other methods are public this should be public as well
databoundconstructor does not belong on constructos of abstract classes
fix miss transport entryservicename and entryserviceid value
add issue reference
apply suggestions from code review co authored by jesse glick jglick cloudbees com
mute indexlevelreplicationtests testseqnocollision signed off by mark vieira portugee gmail com
update copyright year for changed files
add null check for compilation profile
fix graal compiler test base class
handle abstractpointerstamp in duplicatemodified
use primitive boolean
don t append base local path too early
fix sampling bug
tinker bugfix crash leads by conflicts of org apache http library
merge branch master of https github com alibaba druid
documented the descriptors for certain hotspot runtime stubs
rootnode getname upgraded javadoc
fix another issue
ensure all references in infopointreasontest testmethod are resolved
memorycheckpoint memorynode already extends nodeinterface
remove unnecessary overload
add problem source url for simplifyingfractions
script issue fix in rscoredsortedset retainall
fix import in sharedlibraryloader
multiline editor fix macos
fix probability in tailduplicationphase
remove incidental coverage correctly
fix misuse call getconsumerurl
findbugs issue addressed
update reflectioncachesourcecreator java replace printing types to standard out through logging with log level info
increase the potential buffer size in downsampler
feat performance optimization
remove duplicate property in propertycachekey tostring
fix a broken rename refactoring
add get encryptor
move insertsetaddgeneratedkeytoken to rewrite package
avoided synthetic accessor
prevent recursive invocation of size from initializesize
hotfix deprecate abstractyarnclusterdescriptor setdetachedmode and isdetachedmode
object delete in job
remove lucene isempty docidset after the filter ban this method is not used anymore
merge branch master into len
add a check in sprite rotate
rename wasmcompilationbenchmarksuite to wasmmemorycompilationbenchmarksuite
bug fix for wallfilter whitelist lru bug
attribute value extraction error handle fix npe
float and double constants cannot be inlined
fix eclipseformat error
better close logic for concurrent recovery streams
fix test failure
merge remote tracking branch upstream master
add bootstrap mlockall setting to control if mlockall should be called or not
set the disablelocalexecution in executionenvironment to protected
buffer this inputstream out of paranoia guava copy goes byte by byte
add getter and setter
hotfix add rpcutils inf_duration
modify decrementkeygenerator java
remove unused code
remove accidental system out commit
jdbc local transaction revise dotransaction logic
sql editor disable workd under cursor highlight by default
modify method description of democode
fix fastjson autotype
ds selector ui fix
apply dos filter to chart too but relax limit
refined ignoreunresolvableplaceholders javadoc
remove random empty string from test since it triggers a different exception
frameslot fix import
command line parameters fix
hack for some url
merge two switch cases
sulong add tostring to llvmparserresult
log full display name to disambiguate jobs when used with folders
disallow alpha chars in country code field freebie
check that generated node classes exist
do not recycle result if bitmap was reused
allow native image maven plugin multiline buildargs
reorder javadoc at clauses to appease the checkstyle gods
math is hard only do it once
more accurate position for ensurevirtual exception
bug fix for defaultmessagepathbuilder
propagate compilation failure to optimizedcalltarget
fix image loading bug
mark unused newprofilefileextensionconnection as deprecated
agent fix interceptor class about concurrent plugin is incorrect
polish scheduled javadoc add reference to enablescheduling and task annotation driven
mavenconsoleannotator was buffering a line and thus blocking the arrival of the sync marker also remove a duplicate constant
add convenience dynamicobject get object method
java remove unnecessary boxing unboxing for booleans
fix findbugs warning
reduce visibility of test methods
prefer profile photo over system contact photo
update orderby groupby extract handler
fix textfield which i broke because i don t test all the things
fix for counted loop changes
polish forward merge to reflect renamed classes
fix npe in rangeaggregator
ultimatetargetclass falls back to raw class instead of returning null
an empty array is now returned by resolvedjavatype declaredfields for array and primitive types
json readvalue ignores default value
cleaned up warnings in pact common
graalarithmeticstubs defaults to false
fix bug where computed graal version value is longer than the placeholder value
fix minor comment issue in node
truffle repl debugger remove disused main method
skip also androidx super classes
disabling several more tests broken or unstable in marionette
improved immutablecode invariant checking
remove unused specializationnode constructor
make bytearrayencoder sharable
fix grammar in javadoc signed off by radai rosenblatt radai rosenblatt gmail com
deploy group metadata for plugins maven plugin need to populate group level metadata xml with plugin prefix
is from super class false
lazyinitializationtest whitelist optionvalue types
change stringcachestore method names
bulb documenting source code
moveprofiling rename liroptmoveprofiling option to lirprofilemoves
dynamic remove handler without message lost
fix java doc
merge remote tracking branch upstream master
graal test enable metric printing again
remove suppression that were inadvertently merged back in
notate phone number for too many ams
object reference model refactoring add object class info
refactor remove unnecessary declarations in observer pattern
polising fix botched merge
fix failing test from previous commit
improve httprequesthandler error log
htmlunit ignoring a failed test
update sortcomparisonunittest java
bug fix for load filter error
polish code style
workaround for npe on executorservice shutdownnow
add missing param
ora sql fix
merge remote tracking branch origin devel into devel
tag the eventpoller with comment to indicate its experimental status
updating javadoc after struggling to add action to nodes
list selection now width of list not prefwidth
sync servicenameservice starttime bug fix
reduce the visibility of the event json parser
hotfix runtime restore interruption flag in streamrecordwriter close
to meet new onresponse signature return directly
remove unused constructor
relax an assertion that would incorrectly fail for boxed virtual objects
use correct type
fix small javadoc typos for indexshardroutingtable
better cluster stability check
arraycopy don t use checkcast stub if the arrays overlap
fix release build
polish javadoc for transactionconfiguration
register blank watcher when client close
fix error tests producer should wait a while for consumer to be ready
recover sqlparserparameterizedtest checktestcases
fix containskey for a couple maps
sql server query transformers
treat dev and lpt as a cluster
fix void execute methods should not delegate to generic execute methods with a different number of arguments
plus the hash function
allow http and https urls for convenience
add java doc
add more error info
test don t copy write locks or extrafs files
update error info for oracle
fix clear for fieldlookup values
navigator nodes search fix
use after class
corrected error in functionsignatureparser causing test failures
basic javadocs for firefoxoptions
remove formal deprecation from addpropertyvalue just hinting at add now
increase timeout for stomp integration tests
avoid npe in error controller
fix stamp of loweredatomicreadandwritenode
fix test which was has meaningless assertion it seems that small mistake slip due to copy paste
remove unused but instantiated vector2
fix resource leakage in example
make sure sockjssocketbase close is called for raw websockets
resource leak formatter was never closed
next development version
change construction function
add method for basic encoding without padding
fb annotation in sequentialexecutionqueue
small fix to boundingspheredirectionalanalyzer
tree fix getnodeat again
test connection fix do not call on non initialized pages
use loadhub without null check in monitorsnippets
made rtspmethods and rtspresponsestatuses final
fix erroneous test case for allreduce in iterations
hotfix make memorysize comparable
show catalog tables as views
implement tostring for llvmstackpointer
make private collectphis function in llvmbitcodeinstructionvisitor static
use correct index when try to find the type of the message
improved omatpe javadoc
improve javadoc on idletimeout
sort options alphabetically so that diffs do not show false positives
add temporary substitution for system exit
use hasneverinlinedirective instead of canbeinlined in pegraphdecoder
truffle restore opt fail message on compilation failure
add some sanity tests with joda for date time manipulation
add registerfinalizer virtualize
deep copy map when tomap is called
update another failing internal grid test
fix npe in debuggertester when expecteddone was hit too early
implement hassetoptions in compiler optionvaluesimpl for consistency
fix typo in exception message
disable limit by default
table add getpadx y
remove duplicate file separator replacement
fix for eclipse warnings
short circuit isassignablefrom test in arraycopy intrinsification
truffle reverse import reordering change
deleting unused imports
fix check style
remove deprecated usage of optionvalues set in debuggerretrievaltest
om dsl add static to dynamicobject guard where possible
spdy allow multiple notification of closesessionfuture
modify generaldclintegratetest java
restore processing of endpoint meta data
hotfix core fix typo in wrappingproxyutil javadoc
also print the exception message upon failure
add additional test for primitive char conversion
fix dragging inverted horizontal bar chart
merge patch that fixes race with when using event releaser
add fingerprint constructor that takes a list of events
a invoke without a state cannot deoptimize
fix on register value
unittest fix npe if inlining is disabled
remove bundlehost listener on plugin disposal
avoid native transformation of llvmglobal
tests fix test bug in simplevalidatequerytests
change to using textutils
copy graph properties when copying a graph
this wasn t locating interceptor jar it was locating the hudson core instead
use collections emptyiterator in polyglotexceptionimpl
add a test to confirm inlining if size number of call sites is close to limit
refactor unit test name
add cacheable switch to thymeleaf template resolver user can set spring template cache false to change the behaviour
fix typo in javadoc for mergedannotations
sulong add allocglobalnode tostring
no need to populate profile several times in a row it can be populated in the last constructor only
modify the filter
fix bugs in equals and hashcode
foreign keys cache fix
don t remove empty if merge constructs if there are guards depending on the beginnodes
remove unused method
add comments for new config item
update hotel java
dynamicnewarraynode use init_location for runtime call
truffle dsl add some javadoc
remove intermediate materializations when merge has frame state pending bug fix
fix jdbc plugin issue
emulate missing constructors
findbugs wrong warning message formatting in hudson model updatecenter createupdatecenter
executor filtering in view please enter the commit message for your changes lines starting
trigger test again
modify the cat all test case
fix baseline compiler warnings
fix leave group crash
disable staticintspilltest until hsail backend problem is resolved
sulong minor don t add call target dependencies that already exists
improve logger output on startup
add jdk vm to exceptions for the agent
hotfix runtime code clean up in resourceprofile
remove unused import
add setter for defaultpersistenceunitrootlocation this commit adds a setter for the default persistence unit s defaultpersistenceunitrootlocation attribute
add cause onload in case a cause implementation needs to keep a build field
add the maidian
improve documentation for polyglotexception isinterrupted
agnostic inlining method is a compilation method
remove commented code for debugging purposes
tregex escape regex source in compilation unit names
rename to createupdatedeventfornewschema
fix an infinite loop in defaultcompositebytebuf setbytes
remove unused import
fix spacing around switch parentheses
update firstmaster before submitting state update
polishing method access modifier of annotation and interface
aot verify s aheadoftimeverifcationphase aheadoftimeverificationphase g
fix misleading comment
add getselectgroupbyuseridsql into sqlserversqltestsql
replace linkedblockingqueue with linkedtransferqueue in worker pool signed off by radim vansa rvansa redhat com
disable tests on aot
value panel refresh on rsv rows navigation
fix npe if no cap glyphs
temporarily ignore failing subclass inheritance tests
bugfix push constant field value using original kind from field
remove accidently add usings
add import java io objectstreamexception
remove duplicate ignore failing the compilation
remove unused clusterservice member of nodeservice
allow threadgroup in the image heap
add default case for export type when reading the export section
gave compilationtask a tostring method
hotfix table common deprecate tablesource getschema
improve sql wall
java remove disable gpu option for headless chrome
fix compiler errors
fix a small typo in popup error message no databases fond on the server no databases found on the server
project export import fix
fix memory leak in orderedmemoryawarethreadpoolexecutor
fix license headers
remove the slow class isarray call by returning object instead of httpchunk
for check style
a leaf foreign call does not need debug info
preconditions checkstate now throws illegalstateexception instead of illegalargumentexception this will potentiallly break backwards compatibility remove unhelpful message while constructing exceptions
call the parameter initialvalues
remove unnecessary getter
fix missing since tag
modify the test failure
fix checkstyle violation
buffer getstring test
ensure that declaring class of all unsafe accessed fields is reachable
grid listen out for a node being remove in the model
objloader fix deprecated methods
call shard refresh api and not directly the engine one
fix compilation error
fix null pointer in valuecomparator
fix culltest wtf
log intermediate pending searchers as trace instead of debug
remove unused import
fix null target
remove unnecessary comment
session manager sash ratios fix
test that client id can be hyphen or camel cased
remove debug timeout that was forgotten
hotfix kafka add serialversionuid to flinkkafkaproducer010
refactor transactionengine to interface
use mono defaultifempty instead of flux defaultifempty
newer code need not do this
grid add the event bus flags to the distributor server
fix compile error
replacement method for construct initialization
mid tier conditional elimination needs full schedule
modify the cat monthreport
fix the test for deleted passthrough mode option
check for validity before usage
move static method to top of file
fix native image heap writing for hybrid objects
velocity freemarker tilesviewresolver only return a view if the target resource exists now
remove blank line
more strict testing
fix bug in mesh affected indexed rendering with offsetting doh
fix npe in opensslengine
only load term statistics if required
adjustment to unsafe customization macro node
merge branch master of https github com alibaba fastjson
synchronise the list as multiple threads are calling it
fix regression wrt write barriers
remove several instances of system out print
fix test bug
add condition implies
enable security manager by default in tests e g ides otherwise people will be confused when they use maven
don t use a real ucontextregisterdumper in case we re using the llvm backend
introduce cutomization capability for canonicalizer
dbeaver perspective config fix do not show brwser by default
deprecate the old create method
fix wrong assertion
merge remote tracking branch origin devel into devel
if not in jvmci always use service loader to locate abstractpolyglotimpl
add newinstanceserviceloadertest assertloadservice
add null serverconfig getorchestration
increase visibility of temp file handling in test utils
unignoring a passing test for firefox with synthesized events
update channelmanager java
show task create command in tools menu
use provided auto formater
update histofram range
fix getdisplaymodes returns one and only fullscreen mode on android now
add deprecated annotation
remove useless log print
test make sure all shards are allocated before the delete type is being executed
cleanup static member
renamed connection_reset to ignorable_error_message
add callback for drag end
moved writemessagequeue to the util package
switch servlet exception handling from throwable to exception
add documentation to dos
make tailduplicationphase dummyanchornode an anchoringnode
logging unexpected errors in restadapter
remove double masternodetimeout set in restputrepositoryaction
privatization catconstants java construction method
add comment about why transfertointerpreter is used
hotspotsignature getparametertype should not try eager resolving hotspotsignature getreturntype should try resolving again if the cache holds an unresolved type
first wave scheduler contract migration
debughook got miswired before
add parent alpha to fastimage
support logging exceptions in abstracteventbustest
update the javadoc
limit the number of profiled arguments
wait until index is in the clusterstate after restart
fix kmeans example
extend test to check file modification in directory signed off by clement escoffier clement escoffier gmail com
test the ability to disable inlining by flag
fix checkstyle violation
fix npe bug detected by findbugs
fix a buffer leak in binarymemcacheencodertest binarymemcacheencodertest shouldencodecontent did not release one of the three buffers
unsafearraycopysnippets use right kind for boolean case
fix test suite
merge branch dev of https github com sharding sphere sharding sphere into dev new local
temporarily remove file input split assigner from default configuration
hotfix remove unused taskmanagerconfiguration in taskexecutortest
for thread safe use concurrenthashmap on channelthreadexecutorgroup
mocktransportservice doclose assertions should check openconnections under lock
don t use assert class from loader remove the use of assert since it s unavailable that early
run reactive integration tests only as part of performance group
increase whitespace around password logging increase the whitespace to make the password easier to find in the log
more precise marble diagram for the case when zip takes an observable that emits observables
add copyright info
only yield occasionally when serialising the publishing
prevent read register hoisting in createisolate
even if no copy was made writerindex should be advanced
more bug fix
took care of the case where a message implements both httpmessage and httpchunk
fix bullet tests
export wasmtrap as an error
typos corrected in streaminfo
hotfix yarn fix typos in yarnresourcemanager
from super class is false
add check to ensure that the worker was remove from the queue before unsubscribing that worker
fix unsupported exception was not thrown on ashostexception
internalengine add an assert to verify refresh was performed by lucene
take into account br instructions while reading if blocks
add more info about skipped tests to notifications
add java doc
fix write boolean key without quotefieldnames
add missing default label for switch
test only roll over translog if we at least add one operation
make httpclientrequestimpl conn volatile for proper lock obtention
npe in connectionwatchdog fix
protected is a larger scope than package allow it
disable smart commit by default
rlock minor improvements
add support disassembling during codegen
unignoring a test fix in today s firefox nightly
add cause to frameworkexception to expose the actual exception
made word type rewriting do stamp inference so that word typed phis are handled correctly
remove useless methods
exception moved to parent package
unignoring a green test
hotfix runtime code clean ups in activeresourcemanagerfactory
fix checkstyle issues
fix a potential race condition in abstractsocketstringechotest
explicitly build terminal with jansi
fix checkstyle violation
can write json value by raw string
merge remote tracking branch origin master
add some references to class comments
hotfix properly declare complexintegrationtest as an integration test case
add null check for def of other varnodeast in equals method
update adminserviceimpl java it should be this value
fix small bug in androidaudiodevice
fix filterbynodesconstantset in localproperties
missing nullable on response handshake
make framebuffer unbind static
refactor shardingspherealgorithmfactorybean constructor s sequence
fix wrong argument order in writenarrow match rule
fix navigate up behavior for conversations
lost setting of primary term in merge from master
add child filed
fix issue in piechartrenderer
use linkedtransferqueue for omatpe
don t use ioexception apis not in gingerbread
fix remote hdfs logview issue
use this queryresults
moving accessibleobject setaccessible to where the fields are obtained by reflection
pg data type cache made lazy
merge branch master of github com jenkinsci jenkins
even on failure don t return a body for head request
fix vulnerability while deleting backup file
add javadocs for threadcontext
add generic for routecontext sqlstatementcontext
fix unit test error
version info read fix
throw when getminbuffersize returns an error value
hotfix ioutils closequietly closes absolutely quietly
cliregisterer revert the bogus javadoc patch
redissonlock delete fix
flow sensitive less ambitious addfactinstanceof
merge remote tracking branch origin master
polish resourcehttprequesthandler resourceencoder
document attribute alias support in annotationutils
fixes logging usage log line had an extra parameter
update anytoany java
add debug message for job manager shutdown
java using try with resources to close the stream
merge perf fix re enable fast path
qm panel query size limit
add mapper size to plugin list
properly report teh files and size recovered when doing primary local gateway recovery
mute machinelearningit teststartdatafeed
internal close caches after indicesservice
remove unnecessary check
use a sleeping thread for wasting time
secured creds file flags fix
substitutiontype should delegate to substituted type for hasdefaultmethods and declaresdefaultmethods
fix demo app
reduce number of tested sizes in arrayallocationbenchmark
remove redundant symbol
hide attachment keyboard after selecting an action
remove unused import
print env info for demo
mock data text generator fix
rsv ordering fix
fix docs of spritebatch begin end
tests using utc time zone in rangequerybuildertest to make forbidden apis checker happy
fix froyo ioexception bug
arrayequalsnode is a memory access
fix nullconstant equals to handle nullconstant not being a singleton
adapt to changes in master
flow sensitive bug fix registertypeswitchnode
do not escape half constructed call target to root node
allow both yml and yaml as valid yaml configuration file extensions
invalidate job fix disable disconnect on failure
fix minor changes suggested by checkstyle
allow integration sample to download dsl milestone
support oracle jdbc oracledriver
merge branch dev of https github com tencent tinker
back out unneeded imports
made null checks floating
changed lzma compress and decompress methods to be static
protocolconverter does not need an httpclient factory as a static field
field renaming in unsafe load snippet
add fast path for hotspotresolvedobjecttype resolvemethod when the receiver is the declarer of the method
using junit assertions only
new object create error handling
using non abbreviated local variable name
fix findbugs issue
remove senseless testcases from andexpressiontest
disable low grade truffle compilations in substrate for now
fix maven compile error
bitfields corrected autoscroll during drag
modified formatter settings to not insert line break after annotation on parameters
increase width of hashcode field so that columns stay aligned
add a quick shutdown hook
derped when i copied the jar loading code bibl
add size method to pooledlinkedlist
fix rebase debris
fix release build
use standard location for errno h inclusion
ensure com sun jndi dns dnsclient is initialized at runtime as it contains a securerandom instance
test fix rounding issue
customize node action name fix
tunnel invalidate fix
fix check style error
polish configuration javadoc add note that nested configuration classes must be static
fix test failure
remove unused import in o e p pluginmanagerit
callablestatement parameters discover logging fix
error handle fix
reintroduce suppression of deprecation warning
java reusing default wait from base class
procedures duplicate fix
uncommented test case
dequeue inlined nodes
hotfix fix processmemoryutilstestbase testexceptionshouldcontainrequiredconfigoptions to run also for tm
updating ignores on java authentication alert tests
explicitly make shaperenderer implementing disposable
fix a compilation error
add missing dimensions to compilation final fields in llvmblocknode
tinker bugfix add keep annotation to avoid accidental obfuscating
npe fix no active project case
document the processutil class
fix centroids equality tests in tdigeststate equals
replace magic number with named constant
make mergenode simplify work with a null simplifiertool
take into account extra offsets for pie radar charts
foreign keys in generic ddl
sparc changed register counts due to hint
minor java collections work
annotatedlargetext writelogto fix inheritdoc
add close function
test shadowenginetests close translogs on tear down here too
protected against a possible null for lastgc in the previous collection
comments add to rtopic
remove non needed import
source coding is a little complex
properly resize memory during initialization
should read from stdin as the caller might be feeding it from elsewhere
fix a comment
fix spelling error
more tolerance in error message matching
fix a test
remove trailing char
fix method called after upstream refactoring signed off by clement escoffier clement escoffier gmail com
remove empty space
fix typo and a compiler warning
hotfix tests strip completionexceptions in miniclusterclient guardwithsingleretry
enhancer print error exception
sopremoserver checks request for formal validity
deleting deprecated code
print java version in the output of native image version command
no need to catch and re throw threaddeath anymore
add missing javadoc for second parameter
modify allinternaltests java
fix array constructor didn t use ordered parameter nate please double check this
sql completion table finder fix
ensure code compiles with eclipse jdt
increate timeout in kafka tests as windows ci agents are very slow
updated the test to fix a regression
sdk fix bytearraysequence subsequence
clarifying that symlinkpath may not be absolute
adapted junit test to new serialization format
sl another minor framedescriptor fix
add a comment signed off by paulo lopes pmlopes gmail com
adjust tests localdistributedexecutor for low memory settings on build servers
modify class name
hotfix kafka test remove unused statement from testonetooneatleastonce
todo fix world begincontact and endcontact takes a shitload of time at the moment
note that csrf crumbs aren t really needed anymore for api clients
remove shameful system out println
add easy to detect output when compilation failure occurs
no need to run this job on install freebie
fix compile error
proper use max texture units
fix style error
make abstractbeginnode a iterablenodetype and use this in controlflowgraph
made the name for stub call lir ops clearer
fix ogg block too big exception
test fix typo in datehistogramtests that fails the test since it expects dates to be rounded by day
update fullhttpmessage java i think fullhttpmessage httpmessage lasthttpcontent not that fullhttpmessage fullhttpmessage lasthttpcontent
this test is failing randomly so extending the timeout
remove thread sleep in gdx invaders wth
value phi node verify value stamps are compatible
add missing override
pmd fix renamed unit test
gwt timeutils emulation
value view panel update fix
error in type comparison
add cler method
make sure new economy compiler configuration is not picked up as the default
rename orchestratortest at last
avoid amd64kind in hashfunction
enable jni support by default
internal make validationexception methods final and fix javadocs
hotfix ignore empty yarn dynamic_properties when creating configuration
adjust rankevalrequest bwc after backport
delete top navigation
allow the bootup concurrency to be overridden
update ivorytower java
rsv column filters ui fix mongo operations list updated mongo regex support
ops add missing import
hotfix test drop unused closedchannels field in mockinputgate
minor fix of exception string
moving scrolling attributes to glfwscrollcallback
modify alltests java
extending timeout in command concurrency unit test
error thrown when calling clinit differs from opcode to opcode
fix format string in bitmapmemorycachekey tostring
prevent duplication of illegal constant during pea
bug fix for sqlutils format
don t activate ingest on tribe nodes
stage javadoc correction
use java util linkedhashset
sync reactor core scheduler
cat only process error at most time
minor change to use registerbeancomponent instead of directly registering the beandefinition
increased robustness of envelope routing in presence of execution failures
refactor lazy eager loading examples
replace manual min max calculation with math min
modify proxyxamain java
fix wrong import
put string concatenation and identityhashcode behind truffleboundary
test fix explain api awaiting info explanation check
document that property flags is for language specific use signed off by stefan marr git stefan marr de
changed persistent task name to be of similar structure as the others
remove debug logs from imageviewaware
om dsl pause checkstyle for snippets
make typecastnode virtualizable
fix the build
adapt device change to new springcli properties
fix inconsistent style
fix the anchoring node in the lowering phase to not skip over loop exits
fix codacy issues
remove useless check
revert changes in method encode too
avoid unnecessary exception wrapping in packagedprogram invokation
reflection usage fix
new failing test to show that jsonprimitive equals is broken for values that are out of long range
fix log level guard in databasestartupvalidator afterpropertiesset
showing user friendly error message about htmlrunner cli options instead of stacktrace
rename to setcolumnvalue
remove usage of hosted elements in truffle images
test mute morelikethisactiontests artificialdocs
correct some comment of largestlimitedmemorycache the comment which describes largestlimitedmemorycache was the same as usingfreqlimitedmemorycache s
actually set the preference to unregistered
fix dynamichub getprotectiondomain substitution
fullclusterrestartit testrecovery add more info on failure
update mysqlvalidconnectionchecker java cannot resolve com mysq jdbc connection ping method to cannot resolve com mysql jdbc connection ping method
change logging level for circuit breaking to warn
fix missing transfertointerpreter in generated code
changed comment once more to clarify https
fix javadoc of oiodatagramchannelfactory
bug fix for pg sql parser
generic driver schema filters
remove faulty test
clean up warning in gradle build
work on scatterdataset copy
ignore llvminteroptest test059 temporarily
fix occurrences of typo
use resourcelist empty
adjusted minimal supported version
fix java8orearlier location
modify the time sleep
hotfix runtime remove unused method executionjobvertex generatedebugstring
optimize encoding of websocket frames by merge if possible
remove a minor code duplication in the llvm class
note some fixme
add gdxnativesloader uses suffix for shared libs when unpacking them based on system nanotime should solve all issues on lunix
test ensure primaries are allocated before bulk indexing with dymamic mappings
switch on cache of parse statement
escape quotes in quoted arguments
entity editor ui property viewer
fix foreign write of vector values
include command processing fix resource encoding read
accept only directories have config xml file to prevent too many ioexception
hotfix undo removal of clusterclient shutdown for backwards compatibility
fix failing zendiscoveryit
remove memoryawarescheduling option since non memory aware scheduling currently generates invalid code
hotfix tests fix typo in exception message
improve javadoc for source createunavailablesection
unhiding a field
dbeaver statup logging fix include full product version
remove unused method analysis isnostopwords
changed actionresetingpool is public now
clarified repository definition from ddd and remove comparison with dao to reduce confusion
account for parent and deferred registrations in invocationplugins isempty
swinganimator issue swing animation should start from zero
provide debuggertags alwayshalt
add a comment to explicit what drives the animation
add missing copyright header
updated the channelgroup javadoc by remove b releaseexternalresources
marking the selenesebackedwebdriver as deprecated
add missing license headers
hide protocol detail from users
in hystrixcollapser only check if request cache is enabled once
add autotype blacklist
polish javadoc for testtransaction
bug fix for getconnectiontimeoutexception
refactor test case codes
delete blank line
remove left over debugging
ensure time is always updated
inferstamp for compressionnode
add saving subtitle track name for matroskaextractor
don t cache count queries
hotfix tests remove unused methods from failingexecutionvertexoperationsdecorator
java deleting unused imports
provide actionable error message for image path issue
add nonnull annotation to method argument co authored by tobias groza tobigr users noreply github com
add support for getting return value of native windows threads
support pinglobaltxtophysicalconnection setting
fix revolutejoint getlocalanchorb
flow sensitive no precision loss check in downcast
ignore juldelegate test for now signed off by clement escoffier clement escoffier gmail com
interactivebreak should never been committed without being commented out
factory method createsubscription from action0
rename to yamlencryptorruleconfiguration
add callable wrapper to actasync
add hostname of remote receiver to log output
add some function definitions required for selenium emulation
use more modern java api for empty collections
update copyright year
making recently add firefox specific tests to run in local environment only
use case transformer for schema catalog names
updated icense years ci skip
remove some overly verbose log lines
wrong import again
reintroduce suppresswarnings unchecked for parsetoplevelobject
remove star imports
made decal constructor public again because gwt explodes otherwise
add back accidentally remove assertions
same code just un commented lines uncommented lines to populate database in the tes
reenable edge move optimizer and cfg optimizer
hotfix runtime delete unused constructor in retryingregistration
update responsebodyemittercontroller java
add missing mimetypes to smooth streaming manifest parser
delete super renew
pea fix virtualization of checkcastnode check type
use ordinal check on guard stage in loadhubnode lower
so much garbage
add some space between the trace filter and the end of the chain
test add some logging to nodeenvironmenttests testdeletesafe
do not print a warning when library cannot be found
fix a bug in the if fix node deduplication that could lead to deoptimization loops
further since comments not picked up by tool
fix a license error
core don t show rename comment if class name not changed
clarified requirements when overriding lirgeneratortool beforeregisterallocation
ci move velocity log into target directory
map graal gc enum to jdk enum
merge remote tracking branch origin devel into devel
updates after editor feedback
painless remove stale comment and remove declared exception
truffle dsl fix implicit cast optimization bug
fix line length violation
spelling post port signed off by pavel drankov titantins gmail com
remove commented code
inputlistener fix handling mousemoved event
make response class final
it s quite possible that this test causes firefox marionette hang on travis
remove unused import
hotfix tests remove redundant null check
make textfieldmapper fast_phrase_suffix private
merge branch master of https github com libgdx libgdx
corrected programexaminer npe
the future is the other direction freebie
include exception type in detail message for compilation failures
fix postrunnable for lwjglcanvas
simplify because we know winding order
fix unit test
screenshots fix broken screenshot generators
remove faulty constructor
remove th printstacktrace left over
fix newinstance in class has been deprecated
javadoc and add a tostring method
pea reset aliases for proxies
use getplaceholderbciname in hsdebuginfobuilder assert
improve lazy loaded singleton example
hotfix network fix the comments error in nettymessage
value transformation error handle
minor fix to lwjgl target fps oops
fix typo freebie
extract nothing return null
improves a tostring method makes it more readable and deletes an useless semicolon
fix typo in javadoc
change long with bigintger and replace getbook
tinker do not hack mh in app oncreate again to avoid potential dead loop
fix missing al destroy in openalaudio dispose
remove unessesary float
rsv refresh on scrolling is enabled by default
add a convenience method
missed changing a method name reference
rbatch comment add
fix neithor pendingselectedindex nor selectedindex is current after fast scrolling viewpager multiple times
another use of extensionlist lookupsingleton
merge remote tracking branch origin master
have bool json filter extend the base one
fix a buffer leak in binarymemcacheobjectaggregatortest
truffle include slots in framedescriptor tostring
fix malformed urltag javadoc
fix binary compatibility
merge branch ocoanet master
remove the deprecated constructor
parent not jenkins
make method public as is needed for reflection when running in classloader
avoid unnecessary frame allocation in gifdecoder
hotfix minor streamgraphgenerator refactoring
hotfix runtime remove wrong nullable annotation taskdeploymentdescriptor getserializedjobinformation and taskdeploymentdescriptor getserializedtaskinformation cannot return null
move springfailontimeouttests to statements package
deprecated encrypt example with jpa springboot
fix response time
example use renamed image scale type
add tostring method to segment java for debugging purposes
clarify quaternion getaxisangle
remove unnecessary array initialization
refine prevalidate log
another place where we are using getpassword no need to do this
workbench listener fix npe
turn part of assert into real check to avoid array out of bounds
sulong vectortype should not call tostring of its elements for exception message
fix wrong javadoc tag
aggregate numbers formatting
make some fields accessible in effectsclosure
conditional elimination phase during partial evaluation
fix unit test
since apioption is not repeatable using getannotation is sufficient
add thread name to debug info message
color settings typos
hotfix rest add logging to routerhandler
minor cleanup of audiotrack
remove unnecessary override
update improve code style and remove extra whitespace
set the thread status of manually started threads to runnable because no one else will
shorten logging in conversationactivity
use query row
remove unthrown exception
sulong add the failing test s bitcode filename
fix find bugs warnings
suppress serialization warning
make onexpressioncondition package private consistent with the other condition implementations
truffle repl debugger breakpoint list now correctly updated when some are clared
fix npe during busy call
add a comment for the decoding lock
fix minor typo
fix broken formatting
rename to encryptorstrategy
promote jsonreader2 to be the main jsonreader implementation
validationsuite should be executed within wasmtestsuite
hotfix yarn test avoid using mockito for taskexecutorgateway in yarnresourcemanagertest
fix typo in typeinformation
entity editor save fix
fix test assertion line feeds
usagestatisticstest failure java vendor vs java vm vendor
create guardnode instead of fixedguardnode when lowering guardingpinode
use the right property in test
fix formatting violations in sample
app data update
fix apiviz errors
merge remote tracking branch origin master
stop running in example
remove filterednodeiterable distinct
initialise the default interop datalayout when it is add by the runner
add logging helper function for write barrier debugging
use putifabsent instead of put
add a convenience constructor to localaddress
fix primary checkstyle error in javadoc comment
fix hiero clipping glyphs if a glyph is larger than the page size
checking of whitespaces
if during hudson dev run we have updated a bundled plugin honor that pin at next start
let some time to the server close to happen before starting the new server
fix unique call on already gvned constant
hotfix runtime remove accidental reference to hadoop stringutils in checkpointstoragelocationreference
new class description
orchestrationfacade init null
fix fix future join blocking thread
verify pool size against the correct parameter signed off by oded arbel oded geek co il
make fields static where possible
ds set connect time earlier
spreadsheet update panels on value paste
remove unused sysout
add the version number to the json multicast response
change the addition of analysis processors to override the default ones
revert wrong change in inliningutil
add avoidproxy to make things work in saucelabs for our ci
change to a single get call
polish javadoc for ifprofilevalue
use method call instead of value reference so that hystrixthreadpoolproperties may be overridden
changes after review
delimiter redefiner evaluation
make some fields protected
remove last usage of anchor in the front end
remove reference to old request creation
grid column pack fix check item count
suppress warning in servletannotationcontrollerhandlermethodtests
method name change exit exif
fix eclipse warnings
javadoced skin a little
remove pending messages notification
merge branch dev of https github com shardingjdbc sharding jdbc into dev xiaoyu
fix request builder copy constructor add another comment update request tostring
remove a non needed comment
remove unused field
rename variable name
add connection mode
make functionalinterface for testing public
tinker bugfix accidentally triggered fallback to interpret mode after ota on android n o
sentinel address logging add
remove fix seed for version util tests
append the exception during retry to retry log
fix bug in bytecode reconstitution
fix checkstyle error
made compilation id allocation in tests be atomic
don t query contacts db if we have nothing to query freebie
propagate defaultrequestproperties make getdefaultrequestproperties final
pg exluce sequences from schema children
this sentence was already moved to queue task getdefaultauthentication
remove dead isnull nodes in guard lowering phase
fix wrong value increment
oracle mviews default props config fix
partialevaluator quick fix for inlining regression
provide reasoning for the call to truffle getruntime
sql completion value proposals fix
prevent operator chaining directly after iteration nodes
refactored assertion in jsonnodewrapperparameterizedtest shouldcreatedifferentnormalizedkeys to be more readable
use local variable for target class
no interruptedexception here
test coverage for mysqlcominitdbpacket
print rows separately
sulong remove unused members and add override
logging monitor fix
add missing unsafegetlong substitution minor grammar fix a an
fix another two contract violations in abstractchannelbuffer
fix atlas path when relative to tmx file
no need to search the jvmciclassloader path for truffle services
loop transformations limit unrolling to counted loops with a real counted loop exit
worth recording a userinterruption even for the anonymous user
updated stagetest pivot of sprites is in center
canonicalize at least once in iterativecheckcastelimination
fixes a bug when using double types as sorting key
fix java doc
remove non existing link to validglobalcontextwhen
fixup incorrect property name the property in xadatasourceautoconfigurationtests should be data source class name previously the name was missing
mute indexshardtests testrelocatedshardcannotberevivedconcurrently
fix channelfactory already set error
throw an exception when an action is not found
save parameters types
snapshotarray doesn t need to override removeall after all
convert deoptimize to guard also when deoptimize is behind possibly nested loop exit
update copyright year
update iosapplication java
adjust an abbreviation
replace integers checkedcast with built in method
remove f_nocache from linux only used on osx branch
ensure bufutil release is called even if writebytes fails in writeandrelease
typo once two is enough
traceable exception message
enable g newmemoryawarescheduling per default
create a deep copy of the record when changing timestamps
updated audio loader classes to increase subclassing potential
data search save load tree state
fixup field descriptor
how jesse finds all the typos i have no idea
replace var args with single parameter
add a convenience method
change file position
java have grid console permanently redirect to the new ui url
unbreak build freebie
fix malformed javadoc
fix assertion in lir
remove unnecessary debug scope
fix a typo for magfilter assignment in directionalshadowlight
attachment processor remove unused name enum
clarify message on out of order state publish
hotfix runtime remove duplicated check from executiongraph ctor
delete table name
disable separate sql editor connections by default
linelocationtoprobecollectionmap add forget method
fix checkstyle warning for writable
fix casing of test method
use mathutils clamp over manual implementation
rename readaftercheckcast readaftercheckcasttest
remove waiting for green in copytomapperintegrationtests
fix queue might be null
small tweaks no need to call tostring on the string add comments
use a constant value for default master node timeout
update also the constructor with reader
correctly update group title
updated the qotm example so that people do not ask questions about truncated packets
hotfix improve logging in akkarpcactors
add convenient setcolor methods for decal
remove unnecessary substitution annotations
pre initialization should be done without the createprocess privilege
make the cumulation field in bytetomessagedecoder package private to hide it
revent changes to jenkins and jenkinstest
tweaking upgrade test to improve reliability
test use a reproducible source of randomness in shuffle
make guardloweringphase work even when no context can provide an implicitnullchecklimit
truffle fix headers
tracera tracelocalmoveresolver catch instruction insertion problems early
update rliveobjectservice java
add missing runtime natives for tests
didn t mean to change widgetgroup i suck
small cleanup of exception conversion
add check for new ouya device name
fix failure message
remove need for state after on load indexed and new bytecodes
ignore registerpressure flag when building stubs
more test coverage
test avoid npe when zipfile could not be created
fix test method name typos
adjuncts are like static resources and they should be available without read access or else page rendering can fail
ignoring a test in marionette looks like the browser behavior has been changed
loopfragment proxy input may be null
clean up imports
test fix scriptservicetests testfinegrainedsettings that can loop indefinitely
remove string concatenation
fix exception message
add assertnew for mysqlcominitdbpackettest
another formatting fix
last minute data
fix a line length issue in bitmapsubject
correctly get build information
use the more compact nop instructions for alignment
update jadxcliargs java
avoid unnecessary iteration while parsing math flags
provide stub getstubs method to access all installed stubs
implementing java shim for getelementlocationoncescrolledintoview
facepalm now it s just noop code that should be remove
small corrections to mathutil random javadoc
pasting should respect the onlyfontchars setting
enable newinstance lowering by default for all methods
add comment to explain marginfraglistbuilders relation to simplefraglistbuilder
bindactivity move observeon rightside to make it conventional
remove replyaddress from sendasyncresultfailure message
test add ensuregreen to indices created in tophitstests
adjust log message
change objectmap to objectintmap
fix a bug in prioritizedthreadpoolexecutor now execute runnable verifies the command is add as comparable
modify the testcase
add todo to backgroundexceptionevent
fix major bug with sprite rotation
fix issue in selection callback
spaces tabs for impl math
mention matching bean names in exception message in case of non unique entitymanagerfactory lookup
no longer load a recipient for each filter text freebie
nested sql editor context menu fix
do not call expandreadbuffer too often
modify the thead count
java abstract class does not need a public constructor
add a note about httpclientmetrics remaining in connectionmanager
additional javadoc explaining why runnerstack peek might return null
add test for modules
sql keywords addition
add methods to help visualization
fix unit test
fix texturedata wasn t managed
shorten prefix for graalcompileonly matching to org graalvm
fix miscellaneous minor bugs in compiler
fix test cases
behave robustly in the face of errors from administrativemonitor isactivated
gdxpay documentation update for paytest java
add javadoc to filehandle
temporarily disabled lazy task deployment
remove trailing extra curly brace from javadoc see reference
add in those dependencies
marionette unignoring several tests fix in today s nightly
fix link in javadoc
add retry test case for delete by query tests that we retry failed searches scrolls and bulks
remove improper usage of tcp parameter
simplelanguage needs an id property set
deleting obsolete comment
log error if attemptpromise couldn t be set as failed
ignore testloopcompilation test in case immediate compilation is enabled
bugfix do not eliminate exception handler entry blocks
rowid support fix
return correct application context
grid add docker host flag to help command skip ci
specify clusterserializable must be handled by asyncmaps signed off by henri tremblay henri tremblay gmail com
uninherit from abstractcomponent in indexlifecycleservice
add final for shardingspheredatasource
jenkinsbuildsandworkspacesdirectoriestest also has a restart issue it seems
fix npe in volley stream fetcher
fix potential npe before mock check
changed shadoweffect for negative values
do not assign new source section if replacing node already has source section assigned
sql keywords revert
hotfix metric metricregistryimpl log is private
deny checkpoint request from non local host
mute fieldstatsintegrationit testgeopointnotindexed for now
readded a missing import from a merge conflict
cleanup formatting vector2 changes
hotfix table fix typos in tableenvironment javadoc
truffle dsl fix fingbugs error
grid fix redirects test in jettyappservertest
use only ascii characters in the test
navigator context menu fix
fix minor grammatical errors
set a more appropriate since
avoid npe if dailycheck happens to run during shutdown
fix reflective access to compilerimpl
rename to basesqlrewriteengine
use unsafe ensureclassinitialized in hotspotresolvedobjecttype initialize
fix deadlock in analysistype creation
asyncsemaphore tostring method add
add test case for mysqlstringlenencbinaryprotocolvalue
java fix a typo in a test method name
temporarily enable useheapbaseregister by default
for check style
add public getters for frame and register map
ignore libgraal threads in enginesgcedtest
bug fix jetty jndi
hotfix metrics allow queryparameter converters to throw conversionexceptions
move hotspot specific assertion from debuginfobuilder computeframeforstate to hotspotdebuginfobuilder
set thread name for all jobs redis navigator model
polish javadoc of status method on webfluxtags and webmvctags
tinker print details if any new components was found in manifest of new package
relax overly strict assertion
deprecate dynamicobject size isempty
re add trn methods to add translation component to matrix very useful please don t remove
fix bean config
rearrange screen methods to represent lifecycle better
increase maximum probability range in control flow graphs
fix executeupdate returned unexpected row count bug jdbc backend
hotfix use correct classloader in windowoperatormigrationtest
there is no good reason to need the writehead flag
java avoiding use of raw types
sql preview in results panel
use offset when detecting xcontent type
change shardingandmasterslavetogether java
fix npe introduced by change offline note feature
authentication engine for mysql
reenable jvmergonomicstests on windows
util fixnull address javadoc comments and add annotations
fix pactnull serialization
mute internalautodatehistogramtests testreducerandom
actor hasactions to match hasparent etc found it to be common enough
fix compilation errors
switching to acl as
polish javadoc for annotatedelementutils
specify locale when lower casing
fix matrix3 vals is now public
fix backspace key is now properly caught in gwt backend textfield works as intended
fix the import of donotstrip
rsv mode switch fix
remove use of resolvedjavatype tojava from wordtyperewriterphase
merge branch dev into peertube
hotfix fix streamingexecutionfilesinkitcase instability
merge snakeyaml into source
ilc meta read fix inline items
explanation of the latest fix on sslhandler s bad handshake mac exception
remove obsolete comment
entity editor init fix
review remove unnecessary generics
register graalcompilerdirectives infirsttier intrinsic on svm
delayedframemapbuilder remove trackedvirtualstackslot transform
add trustedinterface to list of classes for tests
remove todo for an issue that turned out to be a non issue
add node name to logs
hotfix cassandra fix cassandrasinkbase serialization issue
modify yamlconfigurationexample java
make sure multicast tests pass on all os es
emit llvm code position independent
add java doc for shardingqueryassistedencryptor
write barrier test disable partial unrolling
tinker adjust some logs
add logging initializer to tests
adds padding to drawable width to account for a ninepatch s content area
update grpcchannelmanager java fix wrong log class name
prevent maxicode sampler from sampling off the edge of the image
do not let users add null inputprocessors to multiplexer
waiting for primary terms
refactored cnt into idx for clarification
fix compile errors
fix problem in mockenvironment
changed tostring method of instancetype
test fix groovyscripttests failures
fix merge error
remove unused import
update license header
remove unused code
fix index masking
fix json error from the last commit
add status code checking for schema download
update generated order class
fix test error
tests fix processfailurerecovery tests by aligning restart delay parameter with heartbeat pause
rename jdk internal jvmci to jdk vm ci
do not save ds config on shutdown
bandaid fix for calltreepovider stack trace
fail if lookupdeclaredmethod finds more than one method with the given name and arity
fix a lint warning
fix minor typos in inline documentation
note cause cause is not set by copy constructor co authored by jsoref jsoref users noreply github com
dominator ce remove debug call from info ctor
driver libraries model
bug fix for refactor
fix a javadoc error
fix bubbledataset bug
remove useless codes
fix contact icon tint issues on older android versions
check that hostname was not resolved before sending it to the proxy signed off by alexlehm alexlehm gmail com
ensure targets cleared at the wrong level are actually cleared
bring haslifecycle stop into alignment with stoppable stop
remove package declaration at archaeologistsdilemma
create children cache on demand
refact simpler class statementutil
remove parser call to the cocor parser
change to true
remove unnecessary warning suppression in modelresultmatchers
setting headers to null to remove them is wrong in the reverse proxy handler
merge branch master into master
sl adopt findcontextnode as a child
graphdecoder early detection and better error message for irreducible loops
permission check was missing
replaced for loop with enhanced for
fix in convertdeoptimizetoguardphase for the case where we need to insert a proxynode
include parameter name in exception when value is invalid
do not drop null check in readnode canonical
fix checkstyle issues
increased memory for blockresettableiterators due to higher minbuffersize
add license info
fix nodeparser error synchronization was done too late
fix gwt from remove glcommon i hope
java deleting unused private field
use minigson for deserialization
remove unused imports
allow testcontextmanager to be compiled in eclipse again
fix prevent stack overflow in type inference if update tree is too deep
ensure we test the correct connection instance
pg attribute list read fix
add minimal javadocs for streaminput readgeopoint
fix a bug in abstractbytebuf writezero where the capacity is not auto expanded
deprecate the apache shim
reference alac sample rate bug
bugfix range protection
remove trace signed off by clement escoffier clement escoffier gmail com
add missing warning suppress
fixup button state in test
feature tracera policies add loopratio loopbudget and loopmaxfreq policies
use the executors factory method to create a fix thread pool
fix npe in ghidraapplicationlayout when the layout is used from the ghidradev eclipse plugin
fix a bug where a closed channel was tried to register with the eventloop
fix mainfragment tab icons did not follow theme color
merge remote tracking branch origin devel into devel
dialing down the verboseness
always start with previous xindex to avoid clipping the cubic too soon
print array in assertionerror
disable lastgc disable using lastgc it gives strange results
fail on malformed sort
add mesh calculateboundingbox
fix glframebufferbuilder visibility
display error message if ld gold is missing remove unicode from source file
fix heartbeatalert bug
corrected a typo and improved the english in a master discovery error
alert should be considered a test failure unless overridden by the test case
add scanner in factorail program
tests ensure yellow in simplefacetstests testfilterfacetwithfacetfilterpostmode
adapt to new checkstyle constraints
modify embed testing server
unbreak the compile
temporarily disable circuitbreakerserviceit testparentchecking
improve timeboxing test speed
sl now needs to check for isavailable instead of source null
fix androidzipfilehandle length throwing nullpointerexception in case the file doesn t exists
log output fix
fix wrong compiler hints in jsongenerator
truffle print the source section in the rewrite trace
hotfix javadocs improve javadocs for streamexecutionenvironment addsource
remove system out
fix context null check
align to api change of spring data cassandra
testcorrupttranslogtruncation add logging
propagate espressoexitexception through main node
example delete redundant configuration option
fix handling of sprite vertices array
constantnode methods should not throw internalerror
grid row header tooltips
hotfix metrics extend metricreporterfactory javadocs
fixes missing targetclass variable had to throw another exception to handle null target
remove some older deprecated apis
move to trace level debugging when sigar fails to load
corrected failing test com baeldung derivedquery repository userrepositoryintegrationtest whenbynamelike_thenreturnscorrectresult
add judge whether is localmodel for domainmanager
use parseablebytearray to get alac sample rate
delete result setkeygeneratorcolumnname
sparc compareandswap must not modify the new value
test fix error message assertion
fix for inliningutil dropped outer frame states while inlining
tests merge test timing fix
restoring relevant explanatory comment
copy paste typo back menu
fix wrong comment change
adapted to changes in loggerrule
remove requestid from request events json
add issue reference for todo
temporarily disable constant scheduling again for canonical graph comparisons
no longer support snappy
permit the null key which is written as null
truffle nevervalidassumption can be invalidated
make sure to close writer
zero out memory in between benchmark iterations
merge branch master of https github com wu sheng sky walking git
hotfix runtime make testingcomponentmainthreadexecutor accept componentmainthreadexecutor
add reverse iteration from tail of pooledlinkedlist
fix a potential bug for override args of static method feedback from yunaiv ascrutae
ensure that intrinsic methods are marked as such
fix inline annotation ew
make regexlanguage public so its options are exposed by launchers
fix merge conflict in javadoc
rename falsepredicate to contradictionpredicate
fix wrong content length for websocket handshake messages
modify the task
navigator refresh fix
truffle dsl made guard parsing more robust
implement bswap with implicit size
fix scoping of globals
delete throws sqlexception
update mainmenuunittest java updated the method name to follow bdd conventions
fix warning problem
correct if condition
fix code generation error for debug builds
update observablehttp java remove pesty dangling p
bugfix copy tab builder
simplified flag to use boolean getboolean
graalcompilerstate rename isannotationpresent to ismethodspecannotationpresent
rename to encryptorconfiguration
stop gifs started in gifdrawabletest
merge branch stratosphere
comparing matrix size before content size
fix avoid nullpointerexception
modify unit test
wrong rbatch execution timeout used
fix compilationwatchdog disable per default increase trace interval and number of fatal stack traces
add comment for workaround
clear passphrase field on invalid entry
noting potential issue in consolecommand
needs a new command that escapes windows scm recursive process killing
also add it as a parameter
hotfix fix the access modifiers in taskmanagerrunner
add support for rect join type
bug fix for jsondirect config support
fix buttongroup not remove from the checked buttons list
modify the clientconfigmanager
allow empty symbols
merge branch master of https github com alibaba fastjson
standardmethodsubstitutionstest get valid code for each iteration solves the invalidinstalledcodeexception issue that occurred in case of a deopt
remove insert selectall on orderrepository orderitemrepository
remove duplicated comment
initialize with empty index to allow operation without ever calling initbasedir
add missing child annotation
toarray for existing arrays
provide an example for using introspection in unit tests
add fix that tonative forwards directly to old legacy tonative of both sides are not migrated
verify both agent cripts have been closed
move node ingest up a few lines
sparc use correct position in sparcassembler insertnopaftercbcond
gdxpay bugfix refactoring wasn t quite done yet
remove unnecessary return
fix a javadoc error in channelbuffer
suppressing dead code warning
fix nativecodegenerator ignores gwt inline code now
update config connecting status when zk connection changes
inspect path documentation adjustment the documentation is closer to the documentation on the website
fix javadoc for optionkey mapof
fix up compiler warnings in ringbuffertest
update expectation to match response from anonymous access to health
hotfix move jarlistinfotest to correct package
update license header
changes required after merge
move healthcountsstream subscribe outside synchronized block
test fix indexlookuptests testcallwithdifferentflagsfails
keystores cleanup on restart
sulong style format
use binaryparserexception instead of runtimeexception
add implies interface to logicnode
add r r files types to trufflevm eval uri
java using try with resources to simplify code
replace linkedlist with arraydeque as arraydeque is faster
optimize for empty string case
add link to https jenkins io redirect hudson util secret signed off by nicolas de loof nicolas deloof gmail com
minor naming change
hotfix mesos remove the useless env variable in mesostaskexecutorrunner
servers folder icon id fix
don t show conversation settings for groups
i hope i ve found at last the test that causes other tests to fail
fixes broken license header
log the deploy redeploy error
changing order of actions click wait get data to check not click get wait
changed javadoc regarding backpressure
unignoring a test for ie
create correct stamps in integerdivnode and integerremnode constructors
fix condition check in sockjsclient lifecycle method
remove unused import
widening of this exception incorrectly caught interruptedexception narrowing it down to reduce the harm
change thread name based on download url
refactor router module for checkstyle
some javadoc and moved position of method
fix error message for jvm if it is not available
java linter recommends replacing n with n in string format
updated javadoc for trufflelanguage findmetaobject
remove incorrect structuredgraph istrivial
fix sqlevalvisitorutils lt cast bug
metric for number of implicit null checks in guardloweringphase
remove aggregationcontext cacherecycler
hotfix correct logging statement in resourcemanager
make the firefoxdriver fall back to looking for wires if geckodriver not found
add help text to compiletheworld options
method metrics add missing inlinee scope during inlining
use raw value directly
trove put values
add comment explaining temporarily limited application of snippetframestatecleanupphase
fix redraw issue on gb freebie
hotfix javadoc make first sentence in jobsubmissionexception javadoc end with period
improve docstrings for and of
delete some testing xml and testing functions
forgot to commit this
shaprerenderer x method i used it all the time
monitor logging fix
add a big fat warning for security managers
fix temporary fix for an issue with androidmusic dispose needs more investigation
delete redundant code
fix checkstyle error
om dsl add since tags
add more blacklist
remove unnecessary suppresswarning annotations
for check style
recovery fix potential npe status reference must be not null to fail recovery
cli print correct help for savepoint command
fix foreign varargs calls
disable tests that fail on some host machines
address review feedback
fix test to run on any platform ie is available on windows only
fix full flush when no changes happen in the index so the updated trans id is not written
suppressing test flakiness
postgresql explain plan
polish javadoc for resourcepropertysource
hotfix javadoc fix typo in yarn utils teh the
hotfix correct log statements in yarnresourcemanager
_throw no longer needed
remove obsolete noopsearchrequestbuilder setnostoredfields
remove synchronized block around pathpatternparser remove the synchronized block around pathpatternparser since it is now thread safe
fix slbignumber in unbox
switch reset to experimental to match rxjava
data transfer mappings save fix target table name
fix compile error
catching all exception that may originate from paths get
noting that this class is not really necessary now that cloudbees folder is open source
initial chunk of work on refactoring the gateway syntatic gateway files names commit points that include translog information
show upload progress for view once messages
update user java
add assertion for rulename in givenincorrectorderxml_whenvalidate_thenexpectvalidationerrors
awaited fix a couple of index template tests till we have remove the ingest index template
lru cache on mailmessageidaction messageid
hotfix use correct classloader in continuousfileprocessingmigrationtest
make endpointmbeanexportautoconfiguration public
minimum change needed to get the camera working on glass
skip loadtimeweaverawareprocessor with native images
sulong throw an llvmnativepointerexception on dispatch of an invalid pointer
fix ehcache format
fix shapestats add in shapeprofiler for primitive locations
add a fuzzy relative time styling format
remove explodeloop for method only called in the node constructor
remove todo comment
truffle fix a bug causing compilation not be cleared from the truffle queue
minor fix typo in debug message
had some inappropriately nontransient fields
minor spelling fix
make getstickyevent and removestickyevent generic so the caller doesn t need casts
sql editor query processor reopen fix
delete unused imports in spring test
sql templates page init fix npe fix
add missing symbol for agent protocols
save task action label fix
make nextsequence a local variable again
register index version created in kuromoji unittest since it s a node level setting here
corrected source code comment
even more fun with guardedby
redisson failed to start when one of sentinel servers is down fix
add class name to mbean error message
the defaultcompileroptions singleton can be final
add javadoc comments to observable distinct
fix malformed see javadoc reference
logs write exception to log instead of printing stack trace
fix checkstyle errors
fix bug in new objectserializer
remove forbidden method reference lambda
test expand failure message for wildfly integration tests
use c sample_flag_sync
change default request uri to
don t throw exception from the sample to make sure no ioexception co has to be caught in this case
if not deleting an index don t delete the gateway data
boolean compare ordering switched
fixbug fix registry center connect error
hotfix connectors improve javadocs for singlethreadfetchermanager
improving the context switching related comment
printing server variables
auto refresh classes after renaming
normalize returns with scheduler factories in schedulergroup
trace log rest test headers
fix code ownership and add some javadoc
generic connection page layout fix
fix for implementsinterface
merge branch rc
update copyright headers
refactor method name
improvement from review
add channellocal init
rename hotspotcurrentrawthreadnode currentjavathreadnode
sni pfx test should use a pfxstore
bind specifically to a loopback interface
hotfix log final status and exit code under lock
replace import with explicit imports
adds missing license files
remove autogentokenkey from parsecontext
update dependencybank to next development version
add license header
fix comment grammar
typo error method invoker no longer exists method invoker no longer exists
fix merge error
adjust import package sequence
add debug logging in case classpath is crazy
add niodatagramchannelfactory support to osgi integration
modify the msg size
controlflowanchordirectivetest make if anchored as well
no need to calculate framenumber if only a single frame is there
enable loadclass instrumentation by default
clarified error message when value fields of lirinstruction classes have the wrong modifiers
fix ecj warnings
fix a stamp join to be done the right way stamp joining is not currently commutative
make sieveoferatosthenes as a private method
loredana crusoveanu suggestions shorten the names of the test methods as they re causing a horizontal scroll bar to appear
remove accidental println
remove verbose comment
add textureatlas load textureatlasdata could load multiple textureatlasdatas into a single textureatlas
modify insert set parsing for old parser
add missing vertxgen annotation signed off by paulo lopes pmlopes gmail com
remove final keyword from fixedchannelpool
switched internal setframe setprogress calls this aligns better with the new lottievalueanimator
remove since for non public classes
test remove timeout it s not needed in this test
add proper generic type information to sortedlist constructor
be more lenient with sizes when byte array pool is not full
add problem description for largestprimedivisor
update filescleardatamanualtest java
correct since todo tags
fix for no paradigm
enforce stable option to have help text
add links in javadoc to alternative methods
merge remote tracking branch remotes upstream dev into dev
renamed inner class
more correct coverage in compiled and corner case
rewrote the channelhandlercontext explanation
add missing param documentation
add a logging message that indicates we are done with the boot
remove dead code from getresponse java
symptomatic fix for npe in inliningutil
fix bug where null parameter arrays are passed to the user code
use orchestrationfacade init
documented the reason for the nodeclasssnippets being written
modify class javawritetofiletest
partially fix lwjglawtcanvas it is still broken
make weakxatransactionmanager is singleton
s es recycle es cache recycle
polish containsany of collectionutils by reusing findfirstmatch
hotfix docs update addshutdownhookthread javadocs
rename to repository
pending indicator was not cleared on recycled views freebie
fix the comments
counted loop detection should ignore infopointnodes
modelinstance only set transform if it s not null
hs vm config processor make sure casts are applied to the whole expression
add nullability annotations to getcomputer
truffle cleaned depracated api usage
s javax annotation checkreturnvalue edu umd cs findbugs annotations checkreturnvalue g
make truffle node parent compilationfinal let getencapsulatingsourcesection evaluate at compile time
fix context traversial in shutdown hook
remove unnecessary semicolon in some enum classes
startwith observable marble diagram
increased batch command timeout in hystrixcollapsertest to accommodate travis slowness
tomcat stop process fix
hotfix remove unused methods from minicluster
remove import that should not be here
actually emit dword lea
for java doc
deprecate the old fieldnamingstrategy interface and open up the fieldnamingstrategy2 instead to take its place
fix checkstyle violation
fix typo in transactionmanagementconfigurer javadoc
add missing svn properties
javadoc enhancements for zip iterable zipwith iterable switchover
fix fix synthetic accessor warnings
lirphasesuite add copy
network handlers parser fix
away with the manger time to fix that typo
add channels pipeline channelhandler
test remove unused method
inline transportsearchaction doexecute
don t use shaded apis
check article code matches github
add rx utilities to reactivedialog
remove dead code
make it more clear in the javadocs that you should use later methods to send upstream events from a channeldownstreamhandler
making style fixes and remove usage of junit classes signed off by david marques dpsmarques gmail com
add license to default tvmci files
prevent use of graal with cms collector
prevent a resource leak in the display help handler
fix compilation error
correct default error message
fix bug in file buffers which caused ioexceptions when task is canceled
make debug not require an extra parameter if someone asks for debug they probably want to debug their app
mysql text data type support fix
revert change to replyexception message but provide descriptive tostring method
tests fix test bug foo bar
fix possible npe
delete unused cssescape method
add todo for transactionengine
change minute bug
test wait for yellow else assertsearchresponse will trip
remove addressed todo
add truffle boundary
add safegetxml methods
update loginfilter java
fix bugs in support for arbitrary types in substitutionguard constructors contributed by igor veresov igor veresov oracle com
polish javadoc for contextcache
hotfix runtime minor optimization in checkpointmetrics
hotfix fix checkstyle violations in savepointrestoresettings
typo in comment additional clarifications
don t deprecate old constructors make new ones non public
add setter for polygon s localvertices which are no longer final
add getcursorx y
fix dnd not working after dragging a source without a payload
remove synchronization from imagedecoder
implement tostring in callinstruction
fix import in node field data
fix code style
avoid peing system exit
custom console encoding support
merge branch master of github com jenkinsci jenkins
add a batch insert test case with each insert have multiple into_values
remove deprecated mws method use url instead
hotfix hive make hivecatalog s hiveconf and client wrapper private
better format for simple output
add method fitscreen that resets all dragging and scaling and lets the chart fit exactly on the screen
pass through datasourcepropertymanager
quick fix for threading issue should maybe look into synchronizing this method instead
for check style
prevent a crash when using certain layer types like camera
consider the specified alignment for the alloca instruction
should invoke notifydatachange when show sign point
add javadoc to remove lint warning about unused variable
use math max policy for consolidation of xms arguments
remove useless getter
remove dead code
use simple type name in missing parameter exception text
break add to avoid unnecessary loop iterations get rid of for since a loop is not needed for removeif method
test try to speed up rest tests by reducing max number of replicas and shards
make ismultithreaded more robust
add gettimouet to httprequest
match words final
update batch java
remove some more todos from parsedaggregation
add links to graham lea s useful overview of observeon subscribeon
fix plugin service check for missing descriptor to allow ioexception
changed gl20 field inside iosgraphics to be gl20 instead of iosgles20
remove mario s aids
ignore failing test for now
polish javadoc for transactionaltestexecutionlistener
remove forgotten leftover
move sparcsubintcomparecanonicalizer to lowtier
change order of check since routing entry might be null when the shard is just created
modify the unused jsp
spreadsheet presentation dirty detect fix no ui controls access
a bit of javadoc
test awaitsfix testtriggerupdatesconcurrently
use timed wait to detect idleness only when core threads cannot timeout
don t use thread sleep during the grace period when handling initial debugger command sequence
tracera add suppresswarnings try
make sure cachednontrivialnodecount is invalidated before invalidateinstalledcode returns
run eclipse formatter
hotfix add missing documentation in statetransformationfunction
add jdoc to ssa parseposition method
json use generic types for fromjson
disabled use of unsafe allocateinstance based mechanism for node cloning to measure impact on compile time benchmarks
use correct platformkind in reinterpret lir generation
print the exception when failing to add a specific mapping during cluster state update
fix build break
require that relocation source is marked as relocating before starting recovery to relocation target
test fix test to optimize and flush in the end to try to prevent delete index failed not acked
add setlogger method
fix nodestats comment
support long length in inputlengthnode
add us gov west
hotfix network fix useless backlog value in bufferandavailability returned by remoteinputchannel getnextbuffer
implement get set mousespeed in the rc emulation as a no op
close responsebody in sample
replace identity map with hash map
minor update for disklrucache
arraycopycallnode remove unused method
bugfix join of alwaysnull and nonnull is the empty stamp
codec memcache no need to allocate a buffer if no extra is used
fix issue in which containermediachunk would throw an npe for self initializing chunks
remove scope annotations from dagger modules
add additional checks for version compatibility
update collectionsbenchmark java
fix testcase compiler error
add since for user getbyid to ease the maintenance of plugins avoid searching for first version providing it
system property util might return null motivation isandroid0 should be robust modifications yoda equals for string comparison result no npe
connection page compile fix
java fix logic of isinternetexplorer check
support reading nulls into strings
add canonicalization to interger div
use ellipsis character
tracera add traceallocationphase tostring
fix y being parsed as j
hotfix network add a few more checks and tags
fix dumb test copy paste mistake
hotfix tests fix exception in autoclosableprocess
adds missing copy of ipv6
add simple example to xcontentparser how to obtain an instance of it
ci bump copyright dates
remove override of remove method
handle inaccessible jdk internal vm ci packages in defaulttruffleruntime clinit
conform to java command empty cp entry handling
test mute dedicatedclustersnapshotrestoretests restorepersistentsettingstest
add formatting using locales
take into account basephase subclasses whose name does not end with phase
test reduce the number of fields in mlt tests
xml utils fix
tinker remove redundant code
fix line len
registertask should be a static class
add blank line
returnop ends a block
correct timeout logic when awaiting exception in livereloadservertests
this exception is never thrown
suppress rawtypes warning for createreferencearray
fill before stroke because the fill may cover half of a thick stroke
fix warnings in snippets
tomcatembeddedservletcontainer now fails fast if there are exceptions during protocol handler startup re throw exception
add type for etcdregistrycenter
stop thread interrupt from being called freebie
fix compile error
fix typo in nfastate json debug output
eclipseformat is also applied to package info java files
fix bug in management graph iterator implementation
use correct kind for narrow int constants
fix blacklisted method
throw should not reach here exceptions on invalid jdwp class patterns
transportshardbulkaction add the exception to the message of an assertion about it s type
use implication mechanism in shortcircuitornode
add javadoc to g1referentfieldreadbarrier
buildstep prebuild is to run at the beginning of the build
change ip address provider remove null bomb
fix for objectclonenode
ensure pooledbytebuf init is not called with null memory
integer tostring here too
nodelirbuilder verifyphikind check for lirkind equality first
ignoring recently add test in legacy firefox driver
truffle instrumentation remove instrument related methods from node now implemented by truffellanguage
potential fix for npe in nioworker registertask run
add getframedescriptor to frame interface
add nullable annotation to result of lrucache get
fix delay getactor i m an idiot d
om dsl don t import truffleboundary if it isn t used
fix bug with incorrect output directory preparation
add comment about builtin libraries in espresso
test return the correct transport instance in mock transport
fix java warnings
fix accidental typos sorry
nodeinfo should get the actual port if options had wilcard port signed off by thomas segismont tsegismont gmail com
only show delete action when long pressing on profile change update
merge branch master into ccr
small style fix
recover from one more getsockname crash
move force after multiple writes not at every steps
corrected license header ci skip
modify the cat
clear the percolator reader from the cache explicitly
useridcause getusername and usercause getusername should both create a user if the user does not exist
remove unnecessary logerror
fix broken test
update to the latest reactory netty snapshot
fixes non snapped in range minmaxviewport doh
properly handle guard usages during on stack replacement
make optimizedcalltarget callboundary as lightweight as possible
fix interop error handling
fix pip positioning in video calls
statement delimiter check fix
androidlog add gettag and logchunk methods for easier subclassing
decreased the test loop count
invokenode getannotation is dangerous
lower case tooling cluster
extract nothing return null
documentation that impls must call close on message payloads
on failure to extract a plugin log the message and bail
add deprecated annotation to util getsize
fix build failure
put symbol tostring behind boundary
remove uused import
hotfix tests speed up streaming state tests by skipping default retry delay
lazy compute the hash and actually use it
fix propagate language in debugvalue execute
modify the epoll
rename the remote ip name
remove duplicate logic for request resolution and routing verification
hotfix correct the default path of nvidia gpu discovery
remove unused parameter from compilertovm dontinline
minor javadoc fix signed off by ruslan sennov ruslan sennov gmail com
fix add metadata method npe
eliminatednodenames add props
fix for comparison issue with double
truffle change calltarget from an abstract class to an interface allow varargs
merge branch master of github com redisson redisson
fix typo in giphy urls that breaks the sample app
a workaround for missing algorithm error during handshake
simplify espressorootnode getname
remove unused variable
fix typo in javadoc for testcontextbootstrapper
cleanup remove unused code
improve javadoc of truffleexception regarding fillinstacktrace
avoid use of static nativeimagesystemclassloader field
forgot one local
bugfix use error field
fix wrong test
polishing make runprocesskiller final
validate bufferlimit is positive in heapbufferedasyncresponseconsumer
test muted all hdrpercentilestests and hdrpercentilerankstests
always use at least systemclassloader to load custom message classes
optimizedosrloopnode profilingenabled should propagate to other threads
remove unused import in recoverysourcehandler
enable all permissions in parsertorturesuite
httpclient timeout simulation for a domain
update iosapplicationconfiguration java
fix line width
add documentation on the methods that pick up stamps and change getsafestamp for the primary value
rename to ignoredshardingorchestrationevent
test in percolatortests testpercolatesorting_unsupportedfield we need to be sure that the percolator type exists
hotfix table api java improve documentation of withbuiltincatalogname withbuiltindatabasename
add value accessor to sexpratomnode
add firefoxdriver constructor that accepts driver server only
remove nullcheck for known non null value
test with error instead of runtimeexception test the throwable catch instead of exception
java calling a more straightforward method
rename to assertgeneratekeywithclockcallback
remove trailing comma at alltest java
fix compile issue
fix typo in httpmessageconverters auto conf tests
fix polyglot bindings isexisting message
fix docs for polygonshape thanks tescott
hotfix rocksdb only log snapshot restore message when state is actually restored
loop peeling remember number of peeled iterations per loop
this condition should be opposite
propagating only non internal errors
for check style
simulate false unit test
subscriberinfo clean up
fix spell mistake
fix checkstyle errors
should not attempt to call virtualfile child with the empty string
bug fix for jsonwriter
rename to jpauserservice java
setdelay can set a value that would cause illegalargexception to be thrown by random nextint
does not need to get timeout
test add path home to settings
fix boxing warning
ifnode simplify expression in simplify
sql editor init error handle
make sure we always handle lasthttpcontent messages
remove unnecessary optimize call that destabelizes test
give info if content language is system on crash
remove parsefieldmatcher usages from queryrewritecontext
add output to stderr if we fallback to empty job list
fix memory leak in reactive multipart parser
add javadoc for the categories
use array of correct size when turning list metricreader into an array
remove obsolete suppression of deprecation warning
fix incorrect template name in test case
fix merged code error
only avoid graph caching when the graph was built without profiling information
remove javadoc reference to non existent class in test source code
updated documentation for registertypehierarchyadapter to cover typeadapter
use system out only debug is activated
fix declared patchcontext was not found when generic type argument was used with ecj
minor changes in line renderer
remove deprecated isobjectoflanguage method
truffle dsl fix executable input types do not count as generic if there are implicit casts defined for them
remove condition around the main function being null after parsing a su file
add block test properly
adjust test for new permission changes manage and system read permissions
inliningutil tell which graph in error message
remove unused import
wdbs including original webdriver exception into seleniumexception
fix style and add documentation
merge remote tracking branch origin master
internal cluster name
call ensureinitialized in dynamichub newinstance
code formatted according to checkstyle rules
adapt to the dledger mode
dynamic structs support
fix clean up after pact graph example jobs
remove unused inject
fix an incorrect link in requestoptions diskcachestrategyof
make failedchannelfuture and succeededchannelfuture final
resolve datasource from resource
make prfm accept register_offset addresses
delete blank line
add unit test for student grade
remove redundant code
requestrendering when add an action
window behavior changed so did marble diagram thus its size
lengthen latent commands for semaphore rejection unit test
ensure no memory reads are scheduled before the c to java transition
getitem string itemgroup javadoc failed to mention the fallback to getitembyfullname
fix npe in client homes panel
correction of a runtimeexception uncompilable source code
endpoints final private no arg constructor
fix rx serialization bug in takeuntil again
better error messages at taskmanager startup and registration
fix set of firefox preferences for logging prefs after this fix fxdriver correctly set logging preferences
remove a couple of unused imports in glide s compiler extension test resources
add space for timezones message timezonesby default
fix minor issue in exchangeresult
avoid unnecessary generichttpmessageconverter re declaration
hotfix improve logging in dispatcherresourcemanagercomponent
test use tmp dir otherwise the security manager will complain
fix merge conflict
bookmark create fix
add another option
minor test improvement
hint on downcasting and multicasting
hotfix port the dynamic_properties config option to new options api
remove wrong cglib dependency
fix checkstyle violation
remove unused node intrinsic
core limit region traversal iterations count
throw exception if executequery use unsupported enum
fix phase applications in partial evaluator
made sure oiodatagramchannel joingroup is allowed only when the channel is bound
fix a bug where localserverchannel s localaddress is unregistered
fix fragile tests for bean method overloading
fix ha sync transfer timeout
set the cleanup timer thread of the bloblibrarycachemanager to be a daemon
fix for npe on import
late binding logging fix
add substrateaarch64backendfeature only on lir backend
hotfix move taskmanagerslot to o a f runtime resourcemanager slotmanager
modify the cat close idle time bug
modify the builder
add problem source url for veryeasy
test cleanup static test resources
use linkedhashmap for a deterministic order
fix correct code line number calculation
forgot to remove the todo comment
modify the code
add tostring to compressedstring to show when logging it
add gvn after floatingreadphase
skip unreleased versions in restorebackwardscompatit
sqlutils surpport odps
fix null phase names in igv
fix checkstyle error
reply with error code when source file is not present
mark searchasyncactiontests testfanoutandcollect as awaitsfix
pg default parameters trim
java deleting unused imports
ensure a test of alert dismiss actually calls alert dismiss
rbuckets delete method is deprecated
set the chromedriver log location using a system property
moved test into proper directory
for check style
deleting unused imports
remove unused option
move shardingcondition java
java deleting unused imports
add domainobjectchangerecord event type information
prefix todisplaystring for interop objects with foreign object
remove the getnodefactory from llvmnode
sulong documentation fix
add entry bci in compilationtask tostring
test ensure http is enabled in jsonp tests
truffle dsl add missing override to remove
unignoring a green test
noting that cloudslaveretentionstrategy appears to be a poor reimplementation of cloudretentionstrategy
repair a test case bug
remove add context from engine if context locals initialization fails
modify shardingandmasterslavetogether java
hotfix fix checkstyile violations in bashjavautilstest
implement jni fatalerror
vertxhandler should close using channelhandlercontext instead the channel when it closes upon an exception caught
fix complaints from findbugs
further simplified the vote method
fix apiviz tag errors
merge remote tracking branch origin devel into devel
lenient fail when no rabbit
update replyexception java
made constant field final
only set secure if request is secure
tracera do not modify the block order in bidirectionaltracebuilder
add missing set boolean convenience method
